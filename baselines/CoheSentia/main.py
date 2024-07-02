import os

import sys
import time
import datetime
import random
import argparse
import logging
import gc

import numpy as np
import pandas as pd
import json 

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM,
    AutoModel, 
    AutoConfig, 
    AdamW, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup, 
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,    
    default_data_collator, 
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainer,
    GPT2Tokenizer,
    GPT2ForSequenceClassification
    )

from transformers.trainer_utils import get_last_checkpoint

import transformers
import datasets

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from load_dataset import load_datasets
from utils import print_gpu_utilization, print_summary
from utils import labels_coherence_score, reason_labels, coherent_sent_labels, cohesive_sent_labels, consistent_sent_labels, relevant_sent_labels
from arguments import DataArguments, ModelArguments
from multi_label_model import BertForMultilabelSequenceClassification
from metrics import bert_compute_metrics, t5_compute_metrics, get_prediction_base_thresh
from metrics import get_final_predictions as get_final_predictions_for_t5
from save_predictions import save_pair_bert, save_pair_t5
from mtl_model import MultiTaskModel


# import wandb
# wandb.login()
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_PROJECT"] = "exprmnt"
# os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_WATCH"] = "all"


def set_device():
    # Empty cache of GPU
    torch.cuda.empty_cache()

    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def main(model_args, data_args, training_args, device, extra_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir):

        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (last_checkpoint is not None
              and training_args.resume_from_checkpoint is None):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # loading tokenizer and model
    # config - is per task since there is a different numer of labels for each task
    multi_label = data_args.dataset_type == 'sent_reason'
    mtl_model = data_args.dataset_type == 'mtl'

    if 'gpt' not in model_args.encoder_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.encoder_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        tokenizer=None

    raw_datasets, tasks_info = load_datasets(tokenizer, data_args, training_args, model_args)
    encoder_gradient_checkpointing = extra_args["encoder_gradient_checkpointing"]
    if 'gpt' in model_args.encoder_name_or_path:
        return

    if 't5' in model_args.encoder_name_or_path:
        if coherence_type in ['combined', 'holistic', 'incremental']:
            relevant_labels = labels_coherence_score
        elif coherence_type == 'sent_binary':
            relevant_labels = coherent_sent_labels
        elif coherence_type == 'sent_cohesion':
            relevant_labels = cohesive_sent_labels
        elif coherence_type == 'sent_consistency':
            relevant_labels = consistent_sent_labels
        elif coherence_type == 'sent_relevance':
            relevant_labels = relevant_sent_labels
        else: #coherence_type == 'sent_reason' - multi label
            relevant_labels = reason_labels
        special_tokens = {'additional_special_tokens': relevant_labels}
        for label in relevant_labels:
            if label not in tokenizer.vocab:
                tokenizer.add_special_tokens(special_tokens)
                break
        labels2idx = {label: i for i, label in enumerate(relevant_labels)}
        idx2labels = {v: k for k, v in labels2idx.items()}
        all_labels_tokenized_idx = [tokenizer(l)['input_ids'][0] for l in relevant_labels]
                
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.encoder_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
        # model = AutoModel.from_pretrained(
            model_args.encoder_name_or_path,
            from_tf=bool(".ckpt" in model_args.encoder_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif 'gpt' in model_args.encoder_name_or_path:
        model = GPT2ForSequenceClassification.from_pretrained(
            model_args.encoder_name_or_path,
        )
        # Add a classification head on top of the model
        model.resize_token_embeddings(len(tokenizer))
        model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.encoder_name_or_path,
            num_labels=data_args.num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if mtl_model:
            model = MultiTaskModel(
                model_args.encoder_name_or_path,
                tasks=tasks_info,
                config=config, 
                model_type = get_model_type(model_args.encoder_name_or_path,)
            )
        elif multi_label: 
            model = BertForMultilabelSequenceClassification.from_pretrained(
                model_args.encoder_name_or_path,
                from_tf=bool(".ckpt" in model_args.encoder_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = MultiTaskModel(
                model_args.encoder_name_or_path,
                tasks=[],
                config=config, 
                model_type = get_model_type(model_args.encoder_name_or_path,)
            )
            # model = AutoModelForSequenceClassification.from_pretrained( 
            #             model_args.encoder_name_or_path,
            #             from_tf=bool(".ckpt" in model_args.encoder_name_or_path),
            #             config=config,
            #             cache_dir=model_args.cache_dir,
            #             revision=model_args.model_revision,
            #             use_auth_token=True if model_args.use_auth_token else None,
            #         )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        if ("validation" not in raw_datasets and "validation_matched" not in raw_datasets):
            raise ValueError("--do_eval requires a validation dataset")
        eval_datasets = raw_datasets["validation"]

    if training_args.do_predict:# or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_datasets = raw_datasets["test"]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if 't5' in model_args.encoder_name_or_path:
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding="longest",
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = default_data_collator
 
    if 't5' in model_args.encoder_name_or_path:
        compute_metrics_for_t5 = t5_compute_metrics(tokenizer, data_args, idx2labels, all_labels_tokenized_idx, predict_with_generate=training_args.predict_with_generate)
        trainer = Seq2SeqTrainer(
            # model_init=model,
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_datasets if training_args.do_eval else None,
            compute_metrics=compute_metrics_for_t5, 
            tokenizer=tokenizer,
            data_collator=data_collator
        )
    else:
        compute_metrics_for_bert = bert_compute_metrics(multi_label)
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets if training_args.do_eval else None,
                compute_metrics=compute_metrics_for_bert,
                tokenizer=tokenizer,
                data_collator=data_collator
            )       
    
    print(f"start training")
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        print_summary(metrics)

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()#(output_dir=)  # Saves the tokenizer too for easy upload
        trainer.model.config.to_json_file(training_args.output_dir + 'config.json')

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(f"finish training")


    print(f"start evaluation")
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if 't5' in model_args.encoder_name_or_path:
            num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
            metrics = trainer.evaluate(eval_dataset=eval_datasets, num_beams=num_beams, do_sample=True)
        else:
            metrics = trainer.evaluate(eval_dataset=eval_datasets)
        max_eval_samples = (data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_datasets))
        metrics[f"eval_samples"] = min(max_eval_samples, len(eval_datasets))
        trainer.log_metrics(f"eval", metrics)
        trainer.save_metrics(f"eval", metrics)
        print(f"finish eval")
    

    print(f"start predicting")
    # prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if 't5' in model_args.encoder_name_or_path:
            label_col = "labels"
        else:
            label_col = "label"

        def get_predictions(test_dataset, to_remove:bool = True):
            labels_idx = test_dataset[label_col]
            if to_remove:
                test_dataset = test_dataset.remove_columns(label_col)
            if 't5' in model_args.encoder_name_or_path:
                predictions = trainer.predict(test_dataset, num_beams=num_beams, do_sample=True, metric_key_prefix="predict")
            else:
                predictions = trainer.predict(test_dataset, metric_key_prefix="predict")   
            predictions = predictions.predictions

            if 't5' in model_args.encoder_name_or_path:
                predictions = predictions[0] if isinstance(predictions, tuple) else predictions
                labels_idx = [[(l if l != -100 else 0) for l in label] for label in labels_idx]
                labels_idx = tokenizer.batch_decode(labels_idx, skip_special_tokens=True) 
                predictions = get_final_predictions_for_t5(tokenizer, predictions, all_labels_tokenized_idx, idx2labels, predict_with_generate=training_args.predict_with_generate)
            else:
                predictions = predictions[0] if isinstance(predictions, tuple) else predictions
                logs0 = [pred[0] for pred in predictions]
                logs1 = [pred[1] for pred in predictions]
                if multi_label or data_args.dataset_type in ['combined', 'holistic', 'incremental']:
                    logs2 = [pred[2] for pred in predictions]
                    logs3 = [pred[3] for pred in predictions]
                    logs = [logs0, logs1, logs2, logs3]
                elif data_args.dataset_type in ['combined', 'holistic', 'incremental']:
                    logs4 = [pred[4] for pred in predictions]
                    logs = [logs0, logs1, logs2, logs3, logs4]
                else:
                    logs = [logs0, logs1]
                if multi_label:
                    predictions = get_prediction_base_thresh(predictions)
                else:
                    predictions = np.argmax(predictions, axis=-1)
            return predictions, logs 

        if mtl_model and 't5' not in model_args.encoder_name_or_path:
            for predict_dataset, task in zip(predict_datasets, tasks_info):
                labels_idx = predict_dataset[label_col]
                predict_dataset = predict_dataset.remove_columns(label_col)
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")   
                predictions = predictions.predictions

                logs0 = [pred[0] for pred in predictions]
                logs1 = [pred[1] for pred in predictions]
                logs = [logs0, logs1]
                predictions = np.argmax(predictions, axis=-1)

                if trainer.is_world_process_zero():
                    output_path = os.path.join(training_args.output_dir, 'mtl_model')
                    os.makedirs(output_path, exist_ok = True)
                    output_predict_file = os.path.join(output_path, f"predict_results_{task.name}.txt")
                    save_pair_bert(output_predict_file, predict_dataset, labels_idx, predictions, logs)            

        else:
            labels_idx = predict_datasets[label_col]
            # predict_datasets = predict_datasets.remove_columns(label_col)
            if 't5' in model_args.encoder_name_or_path:
                predictions = trainer.predict(predict_datasets, num_beams=num_beams, do_sample=True, metric_key_prefix="predict")
                # tokenizer.batch_decode(model.generate(**tokenizer([predict_datasets['text'][0]], return_tensors="pt"), num_beams=8, do_sample=True, min_length=1, max_length=10), skip_special_tokens=True)[0]
            else:
                predict_datasets = predict_datasets.remove_columns(label_col)
                predictions = trainer.predict(predict_datasets, metric_key_prefix="predict", )

            predictions = predictions.predictions

            if 't5' in model_args.encoder_name_or_path:
                predictions = predictions[0] if isinstance(predictions, tuple) else predictions
                labels_idx = [[(l if l != -100 else 0) for l in label] for label in labels_idx]
                labels_idx = tokenizer.batch_decode(labels_idx, skip_special_tokens=True) 
                predictions = get_final_predictions_for_t5(tokenizer, predictions, all_labels_tokenized_idx, idx2labels, predict_with_generate=training_args.predict_with_generate)
            else:
                predictions = predictions[0] if isinstance(predictions, tuple) else predictions
                logs0 = [pred[0] for pred in predictions]
                logs1 = [pred[1] for pred in predictions]
                if multi_label or data_args.dataset_type in ['combined', 'holistic', 'incremental']:
                    logs2 = [pred[2] for pred in predictions]
                    logs3 = [pred[3] for pred in predictions]
                    logs = [logs0, logs1, logs2, logs3]
                elif data_args.dataset_type in ['combined', 'holistic', 'incremental']:
                    logs4 = [pred[4] for pred in predictions]
                    logs = [logs0, logs1, logs2, logs3, logs4]
                else:
                    logs = [logs0, logs1]
                if multi_label:
                    predictions = get_prediction_base_thresh(predictions)
                else:
                    predictions = np.argmax(predictions, axis=-1)

            if trainer.is_world_process_zero():
                output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
                if 't5' in model_args.encoder_name_or_path: save_pair_t5(output_predict_file, predict_datasets, labels_idx, predictions)
                else: 
                    save_pair_bert(output_predict_file, predict_datasets, labels_idx, predictions, logs)            

    kwargs = {"finetuned_from": model_args.encoder_name_or_path, "tasks": "text-classification"}
    kwargs["language"] = "en"
    kwargs["dataset_tags"] = "coherence"
    kwargs["dataset_args"] = data_args.dataset_type
    kwargs["dataset"] = f"coherence_{data_args.dataset_type.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return 



def set_training_args(args, output_dir, debug_mode, config, coherence_type, batch_size: int = 3, train_t5: bool = False):
    max_train_samples = config['max_train_samples']
    max_eval_samples = config['max_eval_samples']

    if train_t5:
        generation_max_length = 30 if coherence_type=='sent_reason' else 3
        training_args = Seq2SeqTrainingArguments(
            do_train=not args.only_prediction,
            do_eval=not args.only_prediction,
            do_predict=True,
            output_dir=output_dir,
            learning_rate=config["learning_rate"],
            num_train_epochs=config["epochs"],

            overwrite_output_dir=not args.continue_training,
            resume_from_checkpoint=True if args.continue_training else None,
            per_device_train_batch_size=min(max_train_samples, batch_size) if debug_mode else batch_size,
            per_device_eval_batch_size=min(max_eval_samples, batch_size) if debug_mode else batch_size,
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            label_names=["labels"],
            # remove_unused_columns=False,
            dataloader_drop_last=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            # weight_decay=0.01,
            # save_total_limit=3,
            # load_best_model_at_end=True,
            # report_to="wandb",
            load_best_model_at_end=True,

            predict_with_generate=True,
            # predict_with_generate=False,
            
            generation_max_length=generation_max_length,
            # metric_for_best_model="rouge1" if t5_use_decoder else "loss",
            metric_for_best_model="loss",
            include_inputs_for_metrics=True,

            gradient_accumulation_steps=config["gradient_accumulation_steps"],  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass. default=1     
            gradient_checkpointing=config["gradient_checkpointing"], # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        )
    else:
        training_args = TrainingArguments(
                do_train=not args.only_prediction,
                do_eval=not args.only_prediction,
                do_predict=True,
                output_dir=output_dir,
                learning_rate=config["learning_rate"],
                num_train_epochs=config["epochs"],
                overwrite_output_dir=not args.continue_training,
                resume_from_checkpoint=True if args.continue_training else None,
                per_device_train_batch_size=min(max_train_samples, batch_size) if debug_mode else batch_size,
                per_device_eval_batch_size=min(max_eval_samples, batch_size) if debug_mode else batch_size,
                logging_steps=config["logging_steps"],
                save_steps=config["save_steps"],
                label_names=["labels"],
                # remove_unused_columns=False,
                dataloader_drop_last=False,
                #evaluation_strategy="epoch",
                #remove_unused_columns=False,
                #local_rank=1,
                # weight_decay=0.01,
                # save_total_limit=3,
                # load_best_model_at_end=True,
                # report_to="wandb",

                gradient_accumulation_steps=config["gradient_accumulation_steps"],  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.default=1
                # gradient_accumulation_steps=1,  # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.default=1
                # gradient_checkpointing=True, # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
                fp16=True if torch.cuda.is_available() else False,
            )
    return training_args


def set_data_args(debug_mode, output_dir, data_dir, config, allow_loading: bool = True, coherence_type: str = 'combined'):
    max_seq_length = config["max_seq_length"]
    num_labels = config["num_labels"]
    ignore_pad_token_for_loss = config["ignore_pad_token_for_loss"]

    if debug_mode:
        max_train_samples = config['max_train_samples']
        max_eval_samples = config['max_eval_samples']
        max_predict_samples = config['max_predict_samples']
        data_args = DataArguments(
            max_train_samples=max_train_samples, max_eval_samples=max_eval_samples, max_predict_samples=max_predict_samples,
            max_seq_length=max_seq_length, output_dir=output_dir,
            allow_loading=allow_loading, data_dir=data_dir, 
            dataset_type=coherence_type, 
            num_labels=num_labels,
            num_beams = config["num_beams"]
            )
    else:
        data_args = DataArguments(
            max_seq_length=max_seq_length, output_dir=output_dir,
            data_dir=data_dir, dataset_type=coherence_type, 
            num_labels=num_labels, allow_loading=allow_loading, 
            num_beams = config["num_beams"]
            )
    return data_args


def get_config(coherence_type: str):
    if coherence_type == 'score':
        config_path = 'config_score.json'
    elif coherence_type in ['sent_binary', 'sent_cohesion', 'sent_consistency', 'sent_relevance', 'mtl']:
        config_path = 'config_reason_binary.json'
    elif coherence_type == 'sent_reason': 
        config_path = 'config_reason_multi.json'

    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config


def get_model_type(model_name: str) -> str:
    if "pretrained" in model_name:
        model_type = "bert_base"
    elif "gpt" in model_name:
        model_type = 'gpt'
    elif "span" in model_name:
        if "large" in model_name:
            model_type = "span_large"
        else:
            model_type = "span_base"
    elif "deberta" in model_name:
        if "v2" in model_name:
            model_type = "deberta_v2"
        else:
            model_type = "deberta_v3"
    elif "bert" in model_name:
        if "large" in model_name:
            model_type = "bert_large"
        else:
            model_type = "bert_base"
    elif "flan" in model_name:
        if "base" in model_name:
            model_type = "flan_t5_base"
        elif "small" in model_name:
            model_type = "flan_t5_small"
        else: # large
            model_type = "flan_t5_large"
    else: #t5_v1
        if "base" in model_name:
            model_type = "t5_v1_base"
        elif "small" in model_name:
            model_type = "t5_v1_small"
        else: # large
            model_type = "t5_v1_large"
    return model_type 


def get_parser():
    parser = argparse.ArgumentParser(description='Coherence')
    ## Required parameters 
    parser.add_argument("--output_dir", default="output/", type=str, help="where to save the model")
    parser.add_argument("--data_dir", default="data/", type=str, help="where is the data")
    
    parser.add_argument("--mtl_mode", default=False, help="to train on several tasks or not", type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument("--coherence_type", default="incremental", type=str, help="where the config is located")
    parser.add_argument("--classification_label", default="sent_binary", type=str, help="where the config is located")
    
    # Base Model 
    parser.add_argument("--model_name", default="google-bert/bert-large-uncased", type=str,  help="pretrained model")
    
    # Data Params
    parser.add_argument("--debug_mode", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--only_prediction", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--allow_loading", default=True, type=lambda x: (str(x).lower() == 'true'), help="to debug or not")
    
    parser.add_argument("--batch_size", default=8, type=int, help="the batch size not in debug mode")

    # CUDA Params
    parser.add_argument("--cuda_mode", default=True, type=bool, help="to use GPU if available")
    
    parser.add_argument("--continue_training", default=False, type=lambda x: (str(x).lower() == 'true'), help="To keep training")

    return parser.parse_args()


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = get_parser()

    # device = set_device()
    # device = args.device if (torch.cuda.is_available() and args.cuda_mode) else "cpu"
    device = torch.cuda.current_device() if (torch.cuda.is_available() and args.cuda_mode) else "cpu"
    print("debug mode is {}\n".format(args.debug_mode))
    print("model is {}\n".format(args.model_name))
    print(sys.prefix)

    if args.classification_label == 'score':
        coherence_type = args.coherence_type 
    elif args.mtl_mode:
        coherence_type = 'mtl'
    else:
        coherence_type = args.classification_label

    multilabel_classification = True if args.classification_label == 'sent_reason' else False
    config = get_config(args.classification_label)

    model_args = ModelArguments(encoder_name_or_path=args.model_name)

    if args.debug_mode:
        if args.mtl_mode:
            output_dir_for_data = args.output_dir + args.model_name + '/debug/'
        else:
            output_dir_for_data = args.output_dir + args.model_name + '/debug/' + coherence_type + '/'
        output_dir_for_training = args.output_dir + args.model_name + '/debug/models/' + coherence_type + '/'
    else:
        if args.mtl_mode:
            output_dir_for_data = args.output_dir + args.model_name + '/' 
        else:
            output_dir_for_data = args.output_dir + args.model_name + '/' + coherence_type + '/' 
        output_dir_for_training = args.output_dir + args.model_name + '/models/' + coherence_type + '/'
    
    if args.only_prediction:
        output_dir_for_training += 'only_predictions/'


    # training args
    train_t5 = True if 't5' in args.model_name else False
    training_args = set_training_args(args, output_dir_for_training, args.debug_mode, config, args.classification_label, args.batch_size, train_t5)

    # # start wandb logging
    # wandb_config = {
    #     "encoder_name": args.model_name,
    #     "learning_rate": training_args.learning_rate,
    #     "max_seq_length": config['max_seq_length'],
    #     "epochs": training_args.num_train_epochs,
    # }
    # wandb.init(project="Coherence", name=f"model_name: {args.model_name} for coherence new dataset type {coherence_type}", config=wandb_config, group=args.model_name)

    data_args = set_data_args(args.debug_mode, output_dir_for_data, args.data_dir, config, args.allow_loading, coherence_type)
    extra_args = {'encoder_gradient_checkpointing': config["gradient_checkpointing"]}

    main(model_args, data_args, training_args, device=device, extra_args=extra_args)

    # mark the run as finished
    # wandb.finish()



