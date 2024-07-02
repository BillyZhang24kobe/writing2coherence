import os
import ast
import logging
import pandas as pd
from typing import List

import datasets
from datasets import load_dataset, load_from_disk
from utils import Task, TASK2ID, ID2TASK


logger = logging.getLogger(__name__)


def load_and_cache_coherence_dataset_into_pairs(data_args, tokenizer, training_args, model_name, tasks: List = []):
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

    logger.info("train save path is {train_save_path}")
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
            # train_dataset = train_dataset.to_pandas()
        if os.path.exists(val_save_path):
            logger.info("Loading val dataset from cached file %s", val_save_path)
            validation_dataset = load_from_disk(val_save_path)
            # validation_dataset = validation_dataset.to_pandas()
        if os.path.exists(test_save_path):
            logger.info("Loading test dataset from cached file %s", test_save_path)
            test_dataset = load_from_disk(test_save_path)
            # test_dataset = test_dataset.to_pandas()
        
        num_labels = int(data_args.num_labels)

    if all_files_exist:
        return train_dataset, validation_dataset, test_dataset, num_labels

    logger.info("Creating datasets and saving in cached file")
    
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.data_cache_dir)
    num_labels = data_args.num_labels
    
    max_seq_length = data_args.max_seq_length

    title_prefix = 'Title: '
    text_prefix = 'Text: '
    end_prefix = '\n###\n\n'
    end_compiltion = '\n'
    
    def preprocess_function(examples):
        num_examples = len(examples['title'])
        all_titles = examples['title']    

        if dataset_name in ['combined', 'holistic', 'incremental']:
            texts = examples['text']
            all_labels = examples['label']
            text_prefix = 'Text: '
            start_complition = ' The coherernce score is: '
            question_prefix = 'Task: Give just a discrete coherence score between 1 to 5 for the paragraph as number?'
            prompts = [title_prefix + title + '\n' + text_prefix + text + '\n' + question_prefix + '\n'+ end_prefix for title, text in zip(all_titles, texts)]
            # complitions = [start_complition + str(score + 1) + end_compiltion for score in all_labels]
            complitions = [str(score + 1) + end_compiltion for score in all_labels]
        else:
            prev_prefix = 'Previous Data: '
            new_prefix = 'New Sentence: '
            all_sents = examples['sents']
            prompts, complitions = [], []

        # if dataset_name in ['combined', 'holistic', 'incremental']:
        if dataset_name == 'sent_binary':
            all_labels = examples['coherence_per_sent']
            question_prefix = 'Task: Is the new sentence coherent in regards to the previous data? give a yes or no answer '
        elif dataset_name == 'sent_cohesion':
            all_labels = examples['cohesion_per_sent']
            question_prefix = 'Task: Is the new sentence coheisive in regards to the previous data? give a yes or no answer '
        elif dataset_name == 'sent_consistency':
            all_labels = examples['consistency_per_sent']
            question_prefix = 'Task: Is the new sentence consistent in regards to the previous data? give a yes or no answer '
        elif dataset_name == 'sent_relevance':
            all_labels = examples['relevance_per_sent']
            question_prefix = 'Task: Is the new sentence relevant in regards to the previous data? give a yes or no answer '
        elif dataset_name == 'sent_reason':# - multilabel classification
            all_labels1 = examples['coherence_per_sent']
            all_labels2 = examples['cohesion_per_sent']
            all_labels3 = examples['consistency_per_sent']
            all_labels4 = examples['relevance_per_sent']
            all_labels = [[lab1, lab2, lab3, lab4] for lab1, lab2, lab3, lab4 in zip(all_labels1, all_labels2, all_labels3, all_labels4)]
            question_prefix = 'Task: Is the new sentence coherent, cohesive, consistenct or relevant in regards to the previous data? give a yes or no answer to each item '

        if dataset_name not in ['combined', 'holistic', 'incremental']:
            uid = examples['Unnamed: 0']
            story_titles = []
            pairs_sent1, pairs_sent2, pairs_guid, pairs_labels = [], [], [], []
            for j in range(num_examples):
                title = all_titles[j]
                sents = ast.literal_eval(all_sents[j])
                if not multi_label:
                    labels = ast.literal_eval(all_labels[j])
                    for i, (sent_i, label) in enumerate(labels.items()): 
                        story_titles.append(title)
                        curr_sent = sents[i]
                        prev_sents = ' '.join(sents[:i])
                        pairs_sent1.append(prev_sents)
                        pairs_sent2.append(curr_sent)
                        guid = "%s-%s" % (uid[j], i)
                        pairs_guid.append(guid)
                        pairs_labels.append(int(label))
                        prompt = title_prefix + title + '\n' + prev_prefix + prev_sents + '\n' + new_prefix + curr_sent + '\n' + question_prefix + '\n' + end_prefix
                        complition = ' ' + 'no' if bool(int(label)) else ' yes'
                        prompts.append(prompt)
                        complitions.append(complition)
                else:
                    labels1 = ast.literal_eval(all_labels1[j])
                    labels2 = ast.literal_eval(all_labels2[j])
                    labels3 = ast.literal_eval(all_labels3[j])
                    labels4 = ast.literal_eval(all_labels4[j])
                    for i, (sent_i, label1) in enumerate(labels1.items()): 
                        label2 = labels2[sent_i]
                        label3 = labels3[sent_i]
                        label4 = labels4[sent_i]
                        final_label = [int(label1), int(label2), int(label3), int(label4)]
                        story_titles.append(all_titles[j])
                        curr_sent = sents[i]
                        prev_sents = ' '.join(sents[:i])
                        pairs_sent1.append(prev_sents)
                        pairs_sent2.append(curr_sent)
                        guid = "%s-%s" % (uid[j], i)
                        pairs_guid.append(guid)
                        pairs_labels.append(final_label)
                        prompt = title_prefix + title + '\n' + prev_prefix + prev_sents + '\n' + new_prefix + curr_sent + '\n' + question_prefix + '\n' + end_prefix
                        complition = ' ' + 'no' if bool(int(label)) else ' yes'
                        prompts.append(prompt)
                        complitions.append(complition)
        
        examples_df = pd.DataFrame(list(zip(prompts, complitions)), columns=['prompt', 'completion'])
        examples_dataset = datasets.Dataset.from_pandas(examples_df)
        return examples_dataset


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
                # if dataset_name not in ['combined', 'holistic', 'incremental']:
                train_dataset = preprocess_function(train_dataset)
                if save_mode:
                    train_dataset.save_to_disk(train_save_path)
                    # train_dataset.to_csv(train_save_path2)
                # train_dataset = train_dataset.to_pandas()

        if training_args.do_eval:
            if not (save_mode and os.path.exists(val_save_path)):
                logger.info("Creating validation dataset and saving in cached file %s", val_save_path)
                # if dataset_name not in ['combined', 'holistic', 'incremental']:
                validation_dataset = preprocess_function(validation_dataset)
                if save_mode:
                    validation_dataset.save_to_disk(val_save_path)
                # validation_dataset = validation_dataset.to_pandas()

        if training_args.do_predict:
            if not (save_mode and os.path.exists(test_save_path)):
                logger.info("Creating test dataset and saving in cached file %s", test_save_path)
                # if dataset_name not in ['combined', 'holistic', 'incremental']:
                test_dataset = preprocess_function(test_dataset)
                if save_mode:
                    test_dataset.save_to_disk(test_save_path)
                # test_dataset = test_dataset.to_pandas()    
    return train_dataset, validation_dataset, test_dataset, num_labels

