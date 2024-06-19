import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

data = pd.read_csv('output/gpt/models/gpt-4-turbo/analysis/only_predictions/sent_binary_results.txt', sep='\t')
data = data.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
print(data.head())
print(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)])
data.drop(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)].index, inplace=True)
data = data.astype(int)

p, r, f1, _ = precision_recall_fscore_support(data['real_label'],
                                              data['predicted_label'],
                                              average='macro',
                                              zero_division=1.0)
coherence = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Coherence'])

data = pd.read_csv('output/gpt/models/gpt-4-turbo/analysis/only_predictions/sent_cohesion_results.txt', sep='\t')
data = data.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
print(data.head())
print(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)])
data.drop(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)].index, inplace=True)
data = data.astype(int)

p, r, f1, _ = precision_recall_fscore_support(data['real_label'],
                                              data['predicted_label'],
                                              average='macro',
                                              zero_division=1.0)
cohesion = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Cohesion'])

data = pd.read_csv('output/gpt/models/gpt-4-turbo/analysis/only_predictions/sent_consistency_results.txt', sep='\t')
data = data.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
print(data.head())
print(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)])
data.drop(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)].index, inplace=True)
data = data.astype(int)

p, r, f1, _ = precision_recall_fscore_support(data['real_label'],
                                              data['predicted_label'],
                                              average='macro',
                                              zero_division=1.0)
consistency = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Consistency'])

data = pd.read_csv('output/gpt/models/gpt-4-turbo/analysis/only_predictions/sent_relevance_results.txt', sep='\t')
data = data.replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
print(data.head())
print(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)])
data.drop(data[(data['predicted_label'] != 1) & (data['predicted_label'] != 0)].index, inplace=True)
data = data.astype(int)

p, r, f1, _ = precision_recall_fscore_support(data['real_label'],
                                              data['predicted_label'],
                                              average='macro',
                                              zero_division=1.0)
relevance = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Relevance'])

results = pd.concat([coherence, cohesion, consistency, relevance])
print(results)
