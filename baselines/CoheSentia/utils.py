from pynvml import *
from dataclasses import dataclass
from typing import Dict


labels_coherence_score = [str(i) for i in range(1, 6)]

coherent_sent_labels = ['coherent', 'not_coherent']
cohesive_sent_labels = ['cohesive', 'not_cohesive']
consistent_sent_labels = ['consistent', 'not_consistent']
relevant_sent_labels = ['relevant', 'not_relevant']

reason_labels = []
reason_labels.extend(coherent_sent_labels)
reason_labels.extend(cohesive_sent_labels)
reason_labels.extend(consistent_sent_labels)
reason_labels.extend(relevant_sent_labels)

T5_VERSION_MODELS = ["t5_v1_base", "t5_v1_small", "t5-large", "flan_t5_small", "flan_t5_base", "flan_t5_large"]
BERT_VERSION_MODELS = ["deberta_v2", "deberta_v3", "bert_base", "bert_large"]

TASK2ID = {'sent_cohesion': 0, 'sent_consistency': 1, 'sent_relevance': 2, 'sent_binary': 3}
ID2TASK = {v: k for k, v in TASK2ID.items()}

@dataclass
class Task:
    id: int# a unique task id
    name: str# task name for log prining
    num_labels: int# number of labels


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result['train_runtime']:.2f}")
    print(f"Samples/second: {result['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
