import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from typing import List, Dict, Any
import numpy as np
import csv


CLASSIFICATION_TYPES = ["incremental", "holistic", "sent_binary", "sent_cohesion", "sent_consistency", "sent_relevance"]
# CLASSIFICATION_TYPES = ["sent_binary", "sent_cohesion"]
# MODEL_NAMES = ["text-davinci-003", "gpt-3.5-turbo-0301"]
MODEL_NAMES = ["text-davinci-003"]

ONLY_PRED = False

DATA_PATH = "output/gpt/models/"


def write_analysis(file_path: str, results: List[Dict[str, Any]], coherence_type: str):
    f1 = results[0]
    recall = results[2]
    precision = results[1]
    acc = results[3]
    with open(file_path, 'w') as fout1:
        tsv_writer = csv.writer(fout1, delimiter='\t')
        tsv_writer.writerow([f'coherence results analysis for data {coherence_type}:'])
        tsv_writer.writerow(['accuracy:'])
        tsv_writer.writerow([acc])

        tsv_writer.writerow(['f1:'])
        for eval_method, value in f1.items():
            tsv_writer.writerow([eval_method, value])

        tsv_writer.writerow(['recall:'])
        for eval_method, value in recall.items():
            tsv_writer.writerow([eval_method, value])

        tsv_writer.writerow(['precision:'])
        for eval_method, value in precision.items():
            tsv_writer.writerow([eval_method, value])


def get_data(file_name, coherernce_type: str):
    if coherernce_type in ['incremental', 'holistic']:
        labels = list(range(1,6))
        labels = [str(l) for l in labels]
    else:
        labels = ['yes', 'no']
        labels_dict = {'yes': 1, 'no': 0} # issue: now yes - means this is coherent and no - this is incoherent 
    all_preds, all_golds = [], []
    with open(file_name) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        data = line.split('\t')
        if i == 0:
            for j, d in enumerate(data):
                d = d.split('\n')[0]
                if d == 'predicted_label':
                    pred_idx = j
                if d == 'real_label':
                    gold_idx = j
        else:
            pred = data[pred_idx].split('\n')[0]
            gold = data[gold_idx].split('\n')[0]
            
            if coherernce_type in ['incremental', 'holistic']:
                gold = int(gold)
                if pred.lower() == 'coherence' or pred.lower() == 'co':
                    pred = 5
                elif pred in labels:
                    pred = int(pred)
                elif pred.split('.')[0] in labels:
                    pred = int(pred.split('.')[0])
                elif pred.split('/')[0] in labels:
                    pred = int(pred.split('/')[0])
                else: 
                    print(f"not valid pred in {pred} in line {i}")
                    pred = int(max(labels)) - gold # making sure this is not corrent
            else:
                gold = labels_dict[gold.lower()]
                pred = pred.split(' ')[-1]
                if pred.lower() in ['yes', 'no']:
                    pred = labels_dict[pred.lower()]
                elif pred.lower() == 'n':
                    pred = labels_dict['no']
                else:
                    print(f"not valid pred in {pred} in line {i}")
                    pred = 1 - gold
            all_preds.append(pred)
            all_golds.append(gold)
    return all_golds, all_preds


def extract_metrics(gold: List[int], pred: List[int], labels: List[int]):
    f1, precision, recall = {}, {}, {}
    
    acc = balanced_accuracy_score(gold, pred)

    f1['macro'] = f1_score(gold, pred, average='macro', pos_label=1, labels=labels, zero_division=0)
    f1['weighted'] = f1_score(gold, pred, average='weighted', pos_label=1, labels=labels, zero_division=0)
    f1['micro'] = f1_score(gold, pred, average='micro', pos_label=1, labels=labels, zero_division=0)

    precision['macro'] = precision_score(gold, pred, average='macro', pos_label=1, labels=labels, zero_division=0)
    precision['weighted'] = precision_score(gold, pred, average='weighted', pos_label=1, labels=labels, zero_division=0)
    precision['micro'] = precision_score(gold, pred, average='micro', pos_label=1, labels=labels, zero_division=0)

    recall['macro'] = recall_score(gold, pred, average='macro', pos_label=1, labels=labels, zero_division=0)
    recall['weighted'] = recall_score(gold, pred, average='weighted', pos_label=1, labels=labels, zero_division=0)
    recall['micro'] = recall_score(gold, pred, average='micro', pos_label=1, labels=labels, zero_division=0)

    results = [f1, precision, recall, acc]
    return results


for model_name in MODEL_NAMES:
    print(f"save data for model: {model_name}")
    data_path = DATA_PATH + model_name + "/analysis/" 
    if ONLY_PRED: 
        data_path += 'only_predictions/'
        
    for classification_type in CLASSIFICATION_TYPES:
        print(f"save data for classification: {classification_type}")
        file_name = data_path + classification_type + '_results.txt'
        output_name = data_path + classification_type + '_analysis_results.txt'

        all_golds, all_preds = get_data(file_name, classification_type)
        for only_pos_labels in [False]:
            print(f"running with pos label {only_pos_labels}")
            if classification_type in ["incremental", "holistic"] and only_pos_labels:
                continue
            if only_pos_labels: continue
            if classification_type in ["incremental", "holistic"]:
                labels = list(range(1, 6))
            else:
                labels = [1] if only_pos_labels else None
            results = extract_metrics(all_golds, all_preds, labels)
            write_analysis(output_name, results, classification_type)

