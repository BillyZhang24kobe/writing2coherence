from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from transformers import EvalPrediction
import torch
from collections import defaultdict
from typing import List, Dict, Any

from utils import labels_coherence_score, reason_labels, coherent_sent_labels, cohesive_sent_labels, consistent_sent_labels, relevant_sent_labels


def bert_compute_metrics(multi_label_classification):
    def compute_metrics(eval_pred):
        if multi_label_classification:
            logits, labels = eval_pred
            predictions = get_prediction_base_thresh(logits)
            accuracy = (predictions == torch.from_numpy(labels)).float().mean().item()
            # accuracy = get_prediction_base_thresh(logits, labels)
        else:
            logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
            labels = eval_pred.label_ids
            # predictions = np.argmax(logits, axis=1)
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        recall = recall_score(y_true=labels, y_pred=predictions, average='weighted')
        precision = precision_score(y_true=labels, y_pred=predictions, average='weighted')
        f1 = f1_score(y_true=labels, y_pred=predictions, average='weighted')    
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    return compute_metrics


def get_prediction_base_thresh(y_pred, thresh=0.5, sigmoid=True): 
    y_pred = torch.from_numpy(y_pred)
    # y_true = torch.from_numpy(y_true)
    if sigmoid: 
      y_pred = y_pred.sigmoid()
    predictions = (y_pred>thresh).int()
    # accuracy = (predictions == y_true).float().mean().item()
    return predictions


def t5_compute_metrics(tokenizer, data_args, idx2labels, all_labels_tokenized_idx, predict_with_generate=True):
    def compute_metrics(p: EvalPrediction):
        """
        Input: is 4 arrays
            1. logitc: <batch_size, max_seq_len, vocab_size>
            2. sequence_output: <batch_size, max_seq_len, embed_dim>
            3. pooled_output: <batch_size, embed_dim>
            4. task_id: <batch_size>

        for t5 the input is <batch_size, max_seq_len> - meaning the predicted input_id for each token in the target
        p.predictions contains only logitc 
        """
        predictions, labels, inputs_ids = p.predictions, p.label_ids, p.inputs
        # # logitc, seq_output, pooled_output, tasks_id = predictions
        preds = p.predictions[0] if isinstance(predictions, tuple) else predictions
        # preds_max = np.argmax(preds, axis=-1) # <batch_size, max_seq_len>

        if data_args.ignore_pad_token_for_loss:
            inputs_ids = np.where(inputs_ids != -100, inputs_ids, tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # when generate = False: preds is a tensor: <num_exa, 3, vocab_size> with dtype=float - this is practically logitc
        # when generate = True: preds is a tensor: <num_exa, 3> - this is the pred tokens as label
        decoded_preds = get_final_predictions(tokenizer, preds, all_labels_tokenized_idx, idx2labels, predict_with_generate)
        # if predict_with_generate:
        #     # decoded_preds is the final label
        #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # else:
        #     relevant_logitc = []
        #     for exa_preds in preds:
        #         # relevant_logitc: <num_exa, num_labels>
        #         relevant_logitc.append([exa_preds[0][idx] for idx in all_labels_tokenized_idx])
        #     final_preds_idx = np.argmax(relevant_logitc, axis=1)
        #     decoded_preds = [idx2labels[p] for p in final_preds_idx]

        decoded_inputs = tokenizer.batch_decode(inputs_ids, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        invalid_classification = defaultdict(int)
        metrics = {}
        
        labels_for_metric, preds_for_metric = [], []
        for pred, label in zip(decoded_preds, decoded_labels):
            if pred not in labels_coherence_score:
                invalid_classification[label] += 1
            else:
                labels_for_metric.append(label)
                preds_for_metric.append(pred)
                
        # get final labels 
        decoder_golds, decoder_preds = [], []
        if len(labels_for_metric)>0:
            decoder_preds = [int(label) for label in preds_for_metric]
            decoder_golds = [int(label) for label in labels_for_metric]
    
        # compute accuracy metrices
        metrics = calculate_full_metrices(decoder_preds, decoder_golds)
        metrics['invalid_classification'] = invalid_classification
        # if len(invalid_classification[task_name]):
        #     metrics[task_name]['invalid'] = invalid_classification[task_name]
        metrics = flat_metrics(metrics)
        return metrics
    return compute_metrics


def get_final_predictions(tokenizer, preds, all_labels_tokenized_idx, idx2labels: Dict[int, str], predict_with_generate: bool=False):
    if predict_with_generate:
        # decoded_preds is the final label
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    else:
        relevant_logitc = []
        for exa_preds in preds:
            # relevant_logitc: <num_exa, num_labels>
            relevant_logitc.append([exa_preds[0][idx] for idx in all_labels_tokenized_idx])
        final_preds_idx = np.argmax(relevant_logitc, axis=1)
        decoded_preds = [idx2labels[p] for p in final_preds_idx]
    return decoded_preds


def calculate_full_metrices(preds: List[int], labels: List[int]):
    result = {}
    result['accuracy'] = accuracy_score(preds, labels) if len(preds)>0 else 0
    result['precision_macro'] = precision_score(preds, labels, average='macro') if len(preds)>0 else 0
    result['precision_micro'] = precision_score(preds, labels, average='micro') if len(preds)>0 else 0
    result['precision_weighted'] = precision_score(preds, labels, average='weighted') if len(preds)>0 else 0

    result['recall_macro'] = recall_score(preds, labels, average='macro') if len(preds)>0 else 0
    result['recall_micro'] = recall_score(preds, labels, average='micro') if len(preds)>0 else 0
    result['recall_weighted'] = recall_score(preds, labels, average='weighted') if len(preds)>0 else 0

    result['f1_macro'] = f1_score(preds, labels, average='macro') if len(preds)>0 else 0
    result['f1_micro'] = f1_score(preds, labels, average='micro') if len(preds)>0 else 0
    result['f1_weighted'] = f1_score(preds, labels, average='weighted') if len(preds)>0 else 0
    return result


def flat_metrics(metrics: Dict[str, Dict[str, Any]]):
    final_metrics = {}
    for k, v in metrics.items():
        if k == "invalid_classification":
            for ki, vi in v.items():
                final_k = 'invalid_' + ki
                final_metrics[final_k] = vi
        else:
            final_metrics[k] = v
    return final_metrics