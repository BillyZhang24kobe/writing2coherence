import os
import ast
import logging
import pandas as pd

import datasets
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def load_and_cache_coherence_dataset_into_pairs(data_args, tokenizer, training_args, model_name):
    data_files = {}
    output_dir = data_args.output_dir
    dataset_name = data_args.dataset_type
    multi_label = True if dataset_name == 'sent_reason' else False
    if dataset_name in ['combined', 'holistic', 'incremental']:
        data_files["train"] = data_args.data_dir + f'train_{dataset_name}.csv'
        data_files["validation"] = data_args.data_dir + f'dev_{dataset_name}.csv'
        data_files["test"] = data_args.data_dir + f'test_{dataset_name}.csv'
    else:
        data_files["train"] = data_args.data_dir + 'train_per_sent.csv'
        data_files["validation"] = data_args.data_dir + 'dev_per_sent.csv'
        data_files["test"] = data_args.data_dir + 'test_per_sent.csv'
    extension = data_files["train"].split(".")[-1]

    save_mode = data_args.allow_loading
    all_files_exist = False
    debug_mode = data_args.max_train_samples is not None

    train_save_name = 'cached_{}_{}_{}_{}'.format('train', model_name, str(data_args.max_seq_length), dataset_name)
    val_save_name = 'cached_{}_{}_{}_{}'.format('dev', model_name, str(data_args.max_seq_length), dataset_name)
    test_save_name = 'cached_{}_{}_{}_{}'.format('test', model_name, str(data_args.max_seq_length), dataset_name)  

    train_save_path = os.path.join(output_dir, train_save_name)
    val_save_path = os.path.join(output_dir, val_save_name)
    test_save_path = os.path.join(output_dir, test_save_name)
    
    if save_mode:
        all_files_exist = os.path.exists(train_save_path) and \
                            os.path.exists(val_save_path) and \
                            os.path.exists(test_save_path)

        if os.path.exists(train_save_path): 
            logger.info("Loading train dataset from cached file %s", train_save_path)
            train_dataset = load_from_disk(train_save_path)
        if os.path.exists(val_save_path):
            logger.info("Loading val dataset from cached file %s", val_save_path)
            validation_dataset = load_from_disk(val_save_path)
        if os.path.exists(test_save_path):
            logger.info("Loading test dataset from cached file %s", test_save_path)
            test_dataset = load_from_disk(test_save_path)
        
        num_labels = int(data_args.num_labels)

    if all_files_exist:
        return train_dataset, validation_dataset, test_dataset, num_labels

    logger.info("Creating datasets and saving in cached file")
    
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.data_cache_dir)
    num_labels = data_args.num_labels
    
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    title_prefix = 'title: '
    text_prefix = '. text: '

    def preprocess_function(examples):
        num_examples = len(examples['title'])
        all_titles = examples['title']
        all_sents = examples['sents']
        if dataset_name == 'sent_binary':
            all_labels = examples['coherence_per_sent']
            prefix = "coherent sentence: "
            question_prefix = ' is the new sentence coherent?'
            positive_label = 'yes'
            negative_label = 'no'
        elif dataset_name == 'sent_cohesion':
            all_labels = examples['cohesion_per_sent']
            prefix = "cohesive sentence: "
            question_prefix = ' is the new sentence coheisive?'
            positive_label = 'yes'
            negative_label = 'no'
        elif dataset_name == 'sent_consistency':
            all_labels = examples['consistency_per_sent']
            prefix = "consistent sentence: "
            question_prefix = ' is the new sentence consistent?'
            positive_label = 'yes'
            negative_label = 'no'
        elif dataset_name == 'sent_relevance':
            question_prefix = ' is the new sentence relevant?'
            all_labels = examples['relevance_per_sent']
            prefix = "relevant sentence: "
            positive_label = 'yes'
            negative_label = 'no'
            
        else: # dataset_name == 'sent_reason' - multilabel classification 
            all_labels1 = examples['coherence_per_sent']
            all_labels2 = examples['cohesion_per_sent']
            all_labels3 = examples['consistency_per_sent']
            all_labels4 = examples['relevance_per_sent']
            all_labels = [[lab1, lab2, lab3, lab4] for lab1, lab2, lab3, lab4 in zip(all_labels1, all_labels2, all_labels3, all_labels4)]
            prefix = "multi label incoherence reasons: "

        uid = examples['Unnamed: 0']
        story_titles = []
        pairs_text, pairs_guid, pairs_labels = [], [], []
        for j in range(num_examples):
            sents = ast.literal_eval(all_sents[j])
            title = all_titles[j]
            if not multi_label:
                labels = ast.literal_eval(all_labels[j])
                for i, (sent_i, label) in enumerate(labels.items()): 
                    story_titles.append(title)
                    curr_sent = sents[i]
                    prev_sents = ' '.join(sents[:i])
                    guid = "%s-%s" % (uid[j], i)
                    pairs_guid.append(guid)
                    final_label = negative_label if label else positive_label
                    pairs_labels.append(final_label)
                    pairs_text = prefix + title_prefix + title + 'paragraph: ' + prev_sents + 'new sentence: ' + curr_sent + question_prefix
            else:
                labels1 = ast.literal_eval(all_labels1[j])
                labels2 = ast.literal_eval(all_labels2[j])
                labels3 = ast.literal_eval(all_labels3[j])
                labels4 = ast.literal_eval(all_labels4[j])
                for i, (sent_i, label1) in enumerate(labels1.items()): 
                    label1 = 'not_coherent' if label1 else 'coherent'
                    label2 = 'not_cohesive' if labels2[sent_i] else 'cohesive'
                    label3 = 'not_consistent' if labels3[sent_i] else 'consistent'
                    label4 = 'not_relevant' if labels4[sent_i] else 'relevant'
                    final_label = ' , '.join([label1, label2, label3, label4]) + ' </s>'
                    story_titles.append(all_titles[j])
                    curr_sent = sents[i]
                    prev_sents = ' '.join(sents[:i])
                    guid = "%s-%s" % (uid[j], i)
                    pairs_guid.append(guid)
                    pairs_labels.append(final_label)
                    pairs_text = prefix + title_prefix + title + 'paragraph: ' + prev_sents + 'new sentence: ' + curr_sent
            # prev_sents, curr_sent = ast.literal_eval(all_sents[j])
            # labels = ast.literal_eval(all_labels[j]) # {'sent0': False, 'sent1': True}
            # story_titles.append(all_titles[j])
            # guid = "%s-%s" % (uid[j], 0)
            # pairs_guid.append(guid)
            # final_label = negative_label if labels['sent1'] else positive_label
            # pairs_labels.append(final_label)
            # pairs_text.append(prefix + title_prefix + all_titles[j] + 'paragraph: ' + prev_sents + 'new sentence: ' + curr_sent + question_prefix)
        
        examples_df = pd.DataFrame({'guid': pairs_guid,
                                    'title': story_titles,
                                    'text': pairs_text,
                                    'label': pairs_labels})
        examples_dataset = datasets.Dataset.from_pandas(examples_df)
        return examples_dataset


    def tokenize_function(examples):
        texts = examples['text']
        labels = examples['label']
        titles = examples['title']
        
        if dataset_name in ['combined', 'holistic', 'incremental']:
            labels = [str(label+1) + ' </s>' for label in labels]
            prefix = 'coherence score: '
            question_prefix = ' what is the coherence score (between 1 to 5)?'
            texts = [prefix + title_prefix + title + text_prefix + text + question_prefix for text, title in zip(texts, titles)]
        else:
            labels = [label + ' </s>' for label in labels]

        # Tokenize
        model_input = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        with tokenizer.as_target_tokenizer():
            tokenized_labels_examples = tokenizer(
                labels,
                truncation=True,
                max_length=30 if multi_label else 3,# the number of tokens for a number is 2 
                # padding="max_length" if data_args.pad_to_max_length else False,
            )
            tokenized_labels = tokenized_labels_examples["input_ids"]
            if data_args.pad_to_max_length:
                tokenized_labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in tokenized_labels]

            model_input["labels"] = tokenized_labels
            model_input["label_mask"] = tokenized_labels_examples["attention_mask"]
        return model_input


    if not (save_mode and os.path.exists(train_save_path)):
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if not (save_mode and os.path.exists(val_save_path)):
        validation_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            validation_dataset = validation_dataset.select(range(data_args.max_eval_samples))
    if not (save_mode and os.path.exists(test_save_path)):
        test_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))
    
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        if training_args.do_train:
            if not (save_mode and os.path.exists(train_save_path)):
                logger.info("Creating train dataset and saving in cached file %s", train_save_path)
                if dataset_name not in ['combined', 'holistic', 'incremental']:
                    train_dataset = preprocess_function(train_dataset)
                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                train_dataset = train_dataset.remove_columns(["label"])
                if save_mode:
                    train_dataset.save_to_disk(train_save_path)
                    # train_dataset.to_csv(train_save_path2)
                # train_dataset = train_dataset.to_pandas()

        if training_args.do_eval:
            if not (save_mode and os.path.exists(val_save_path)):
                logger.info("Creating validation dataset and saving in cached file %s", val_save_path)
                if dataset_name not in ['combined', 'holistic', 'incremental']:
                    validation_dataset = preprocess_function(validation_dataset)
                validation_dataset = validation_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                validation_dataset = validation_dataset.remove_columns(["label"])
                if save_mode:
                    validation_dataset.save_to_disk(val_save_path)
                # validation_dataset = validation_dataset.to_pandas()

        if training_args.do_predict:
            if not (save_mode and os.path.exists(test_save_path)):
                logger.info("Creating test dataset and saving in cached file %s", test_save_path)
                if dataset_name not in ['combined', 'holistic', 'incremental']:
                    test_dataset = preprocess_function(test_dataset)
                test_dataset = test_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
                test_dataset = test_dataset.remove_columns(["label"])
                if save_mode:
                    test_dataset.save_to_disk(test_save_path)
                # test_dataset = test_dataset.to_pandas()    
    return train_dataset, validation_dataset, test_dataset, num_labels

