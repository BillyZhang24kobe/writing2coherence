import pandas as pd
import os
import numpy as np
import json
from typing import List


MODEL_NAMES = [
    "bert-base-cased", "bert-large-uncased", "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", 
    "google/t5-v1_1-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large", 
    "gpt-3.5-turbo-0301", "text-davinci-003"
    ]
DATA_PATH = 'output/' 

CLASSIFICATION_TYPES = ["score", "sent_binary", "sent_cohesion", "sent_consistency", "sent_relevance"]
GPT_ONLY_PRED = False

def model_to_name(model_name: str) -> str:
    if 'deberta' in model_name:
        final_name = 'deberta'
    elif 'bert' in model_name:
        final_name = 'bert'
    elif 'flan' in model_name:
        final_name = 'flan_t5'
    elif 't5' in model_name:
        final_name = 't5'
    elif 'gpt' in model_name:
        final_name = 'chatgpt'
    elif 'davinci' in model_name:
        final_name = 'gpt_davinci'
    
    if 'base' in model_name:
        final_name += '_base'
    elif 'large' in model_name:
        final_name += '_large'
    return final_name


def get_metrics(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    accuracy = int(round(float(lines[2].split('\n')[0]), 2) * 100)
    f1_macro = int(round(float(lines[4].split('\n')[0].split('\t')[-1]), 2) * 100)
    f1_weighted = int(round(float(lines[4].split('\n')[0].split('\t')[-1]), 2) * 100)
    f1_micro = int(round(float(lines[6].split('\n')[0].split('\t')[-1]), 2) * 100)
    recall_macro = int(round(float(lines[8].split('\n')[0].split('\t')[-1]), 2) * 100)
    recall_weighted = int(round(float(lines[9].split('\n')[0].split('\t')[-1]), 2) * 100)
    recall_micro = int(round(float(lines[10].split('\n')[0].split('\t')[-1]), 2) * 100)
    precision_macro = int(round(float(lines[12].split('\n')[0].split('\t')[-1]), 2) * 100)
    precision_weighted = int(round(float(lines[13].split('\n')[0].split('\t')[-1]), 2) * 100)
    precision_micro = int(round(float(lines[14].split('\n')[0].split('\t')[-1]), 2) * 100)
    return accuracy, f1_macro, f1_weighted, f1_micro, recall_macro, recall_micro, recall_weighted, precision_macro, precision_weighted, precision_micro


def get_only_accuracy(file_path: str):
    with open(file_path) as f:
        lines = f.readlines()
    accuracy = round(float(lines[2].split('\n')[0]), 2)
    return accuracy


def save_data_in_json(model_names: List[str], data_path: str, classification_type: str, only_pos_labels: bool = False):
    get_accuracy_dict = len(model_names) > 1
    save_path = data_path + 'results/' 
    if get_accuracy_dict and classification_type != 'score':
        accuracy_dict = {}
        for model_name in model_names:
            if not GPT_ONLY_PRED and model_name == "gpt-3.5-turbo-0301": 
                continue
            model_name_to_save = model_to_name(model_name)
            if 'gpt' in model_name_to_save:
                load_path = data_path + 'gpt' + '/models/' + model_name + '/analysis/'
                if GPT_ONLY_PRED:
                    load_path += 'only_predictions/'
            else:
                load_path = data_path + model_name + '/models/analysis/'
            file = load_path + classification_type + '_analysis_results.txt'
            accuracy = get_only_accuracy(file)
            accuracy_dict[model_name_to_save] = accuracy
        if GPT_ONLY_PRED and 'gpt' in model_name_to_save:
            save_path += 'only_predictions/'
        save_path += classification_type + '/'
        os.makedirs(save_path, exist_ok=True)
        with open(save_path + "accuracy_results.json", "w") as outfile:
            json.dump(accuracy_dict, outfile)
        
    else:
        model_name = model_names[0]
        model_name_to_save = model_to_name(model_name)
        if 'gpt' in model_name_to_save:
            load_path = data_path + 'gpt' + '/models/' + model_name + '/analysis/'
            if GPT_ONLY_PRED:
                load_path += 'only_predictions/'
        else:
            load_path = data_path + model_name + '/models/analysis/'
        if classification_type == 'score':
            file1 = load_path + 'incremental_analysis_results.txt'
            accuracy1, f1_macro1, f1_weighted1, f1_micro1, recall_macro1, recall_micro1, recall_weighted1, precision_macro1, precision_weighted1, precision_micro1 = get_metrics(file1)
            file2 = load_path + 'holistic_analysis_results.txt'
            accuracy2, f1_macro2, f1_weighted2, f1_micro2, recall_macro2, recall_micro2, recall_weighted2, precision_macro2, precision_weighted2, precision_micro2 = get_metrics(file2)

            precision_macro_dict = {'holistic': precision_macro2, 'incremental': precision_macro1}
            precision_micro_dict = {'holistic': precision_micro2, 'incremental': precision_micro1}
            precision_weighted_dict = {'holistic': precision_weighted2, 'incremental': precision_weighted1}
            
            recall_macro_dict = {'holistic': recall_macro2, 'incremental': recall_macro1}
            recall_micro_dict = {'holistic': recall_micro2, 'incremental': recall_micro1}
            recall_weighted_dict = {'holistic': recall_weighted2, 'incremental': recall_weighted1}
            
            f1_macro_dict = {'holistic': f1_macro2, 'incremental': f1_macro1}
            f1_micro_dict = {'holistic': f1_micro2, 'incremental': f1_micro1}
            f1_weighted_dict = {'holistic': f1_weighted2, 'incremental': f1_weighted1}

            accuracy_dict = {'holistic': accuracy2, 'incremental': accuracy1}

            macro_dict = {'precision': precision_macro_dict, 'recall': recall_macro_dict, 'f1': f1_macro_dict, 'accuracy': accuracy_dict}
            micro_dict = {'precision': precision_micro_dict, 'recall': recall_micro_dict, 'f1': f1_micro_dict, 'accuracy': accuracy_dict}
            weighted_dict = {'precision': precision_weighted_dict, 'recall': recall_weighted_dict, 'f1': f1_weighted_dict, 'accuracy': accuracy_dict}

            results_dict = {'macro': macro_dict, 'micro': micro_dict, 'weighted': weighted_dict}

            if GPT_ONLY_PRED and 'gpt' in model_name_to_save:
                save_path += 'only_predictions/'
            save_path += 'coherence_score/'
            os.makedirs(save_path, exist_ok=True)
            with open(save_path + model_name_to_save + "_results.json", "w") as outfile:
                json.dump(results_dict, outfile)

        if classification_type in ['sent_binary', 'sent_cohesion', 'sent_consistency', 'sent_relevance']:
            if 'gpt' not in model_name_to_save and only_pos_labels:
                load_path += 'only_pos_label/'
            file = load_path + classification_type + '_analysis_results.txt'
            accuracy, f1_macro, f1_weighted, f1_micro, recall_macro, recall_micro, recall_weighted, precision_macro, precision_weighted, precision_micro = get_metrics(file)
            
            macro_dict = {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro, 'accuracy': accuracy}
            micro_dict = {'precision': precision_micro, 'recall': recall_micro, 'f1': f1_micro, 'accuracy': accuracy}
            weighted_dict = {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted, 'accuracy': accuracy}

            results_dict = {'macro': macro_dict, 'micro': micro_dict, 'weighted': weighted_dict}

            if only_pos_labels:
                save_path += 'only_pos_label/'
            if GPT_ONLY_PRED and 'gpt' in model_name_to_save:
                save_path += 'only_predictions/'
            save_path += classification_type + '/'
            os.makedirs(save_path, exist_ok=True)
            with open(save_path + model_name_to_save + "_results.json", "w") as outfile:
                json.dump(results_dict, outfile)
    

for classification_type in CLASSIFICATION_TYPES:
    print(f"save data for classification: {classification_type}")
    for only_pos_labels in [False, True]:
        print(f"running with pos label {only_pos_labels}")
        if classification_type == "score" and only_pos_labels:
            continue
        for model_name in MODEL_NAMES:    
            if not GPT_ONLY_PRED and model_name == 'gpt-3.5-turbo-0301':
                continue
            print(f"save data for model: {model_name}")
            save_data_in_json([model_name], DATA_PATH, classification_type, only_pos_labels)
    print(f"save accuracy data for classification")
    save_data_in_json(MODEL_NAMES, DATA_PATH, classification_type)

