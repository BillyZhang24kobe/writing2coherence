import argparse
import os
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
import csv


def read_txt(filename, label2idx):
    data = []
    with open(filename, "r") as inp:
        lines = inp.readlines()
    for i, line in enumerate(lines):
        line = line.split('\t')
        line[-1] = line[-1].split('\n')[0]
        if i==0:
            for l_idx, l in enumerate(line):
                if l == 'real-label':
                    gold_idx = l_idx
                if l == 'prediction-idx':
                    pred_idx = l_idx
        if i > 0:
            try:
                gold = int(line[gold_idx])
                pred =int(line[pred_idx])
            except:
                gold = line[gold_idx].split(' ')[0]
                gold = label2idx[gold]
                pred = line[pred_idx].split(' ')[0]
                pred = label2idx[pred]
            data.append((pred, gold))
    return data


def extract_metrics(data: List, only_pos_labels: bool = False, num_labels: int = 5):
    f1, precision, recall = {}, {}, {}
    gold = [g for p,g in data]
    pred = [p for p, g in data]
    
    labels = [1] if only_pos_labels else None
    acc = balanced_accuracy_score(gold, pred)
    
    f1['macro'] = f1_score(gold, pred, average='macro', pos_label=1, labels=labels)
    f1['weighted'] = f1_score(gold, pred, average='weighted', pos_label=1, labels=labels)
    f1['micro'] = f1_score(gold, pred, average='micro', pos_label=1, labels=labels)

    precision['macro'] = precision_score(gold, pred, average='macro', pos_label=1, labels=labels)
    precision['weighted'] = precision_score(gold, pred, average='weighted', pos_label=1, labels=labels)
    precision['micro'] = precision_score(gold, pred, average='micro', pos_label=1, labels=labels)

    recall['macro'] = recall_score(gold, pred, average='macro', pos_label=1, labels=labels)
    recall['weighted'] = recall_score(gold, pred, average='weighted', pos_label=1, labels=labels)
    recall['micro'] = recall_score(gold, pred, average='micro', pos_label=1, labels=labels)

    results = [f1, precision, recall, acc]
    return results


def write_analysis(file_path, results, coherence_type):
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

def get_label2idx(coherence_type):
    if coherence_type == 'sent_binary':
        labels = ['coherent', 'not_']
    elif coherence_type == 'sent_cohesion':
        labels = ['cohesive', 'not_']
    elif coherence_type == 'sent_consistency':
        labels = ['consistent', 'not_']
    elif coherence_type == 'sent_relevance':
        labels = ['relevant', 'not_']
    else:
        labels = list(range(5))

    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i:l for l, i in label2idx.items()}

    return label2idx

def get_parser():
    parser = argparse.ArgumentParser(description='MTL')
    ## Required parameters 
    parser.add_argument("--data_path", default="output_cohesion/", type=str, help="where is the data")

    parser.add_argument("--coherence_type", default="sent_cohesion", type=str, help="where the config is located")
    parser.add_argument("--model_name", default="google-bert/bert-base-uncased", type=str, help="where the config is located")

    parser.add_argument("--only_pos_labels", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--only_prediction", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument("--to_save", default=True, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    data_path = os.path.join(args.data_path, args.model_name) + '/models/' 
    data_file = data_path + args.coherence_type
    if args.only_prediction:
        data_file += '/only_predictions'
    data_file += '/predict_results.txt'
    label2idx = get_label2idx(args.coherence_type)
    data = read_txt(data_file, label2idx)
    num_labels = 5 if args.coherence_type in ["incremental", "holistic"] else 2
    results = extract_metrics(data, args.only_pos_labels, num_labels)

    output_path = data_path + 'analysis/' 
    if args.only_prediction:
        output_path += 'only_predictions/'
    if args.only_pos_labels and args.coherence_type not in ["incremental", "holistic"]:
        output_path += 'only_pos_label/'
    os.makedirs(output_path, exist_ok=True)
    output_path += args.coherence_type + '_analysis_results.txt'

    if args.to_save:
        write_analysis(output_path, results, args.coherence_type)


