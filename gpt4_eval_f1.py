import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

num_shot = 16

with open(f'output/AllDataGPT_{num_shot}shot.json') as f:
    data = json.load(f)

for entry in data:
    if 'edit_gpt' not in entry:
        entry['edit_gpt'] = []

y_true_coh = [entry['is_coherent'] for entry in data]
y_pred_coh = [entry['is_coherent_gpt'] for entry in data]

y_true_fine, y_pred_fine = {}, {} # R1, R2, ..., R7
for reason_idx in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'):
    y_true_fine[reason_idx] = [not any(reason['reason'][:2] == reason_idx for reason in entry['edit']) for entry in data]
    y_pred_fine[reason_idx] = [not any(reason['reason'][:2] == reason_idx for reason in entry['edit_gpt']) for entry in data]

y_true_cat, y_pred_cat = {}, {} # Cohesion, Consistency, Relevance
y_true_cat['Cohesion'] = [r1 and r2 and r3 for r1, r2, r3 in zip(y_true_fine['R1'], y_true_fine['R2'], y_true_fine['R3'])]
y_pred_cat['Cohesion'] = [r1 and r2 and r3 for r1, r2, r3 in zip(y_pred_fine['R1'], y_pred_fine['R2'], y_pred_fine['R3'])]
y_true_cat['Consistency'] = [r4 and r5 for r4, r5 in zip(y_true_fine['R4'], y_true_fine['R5'])]
y_pred_cat['Consistency'] = [r4 and r5 for r4, r5 in zip(y_pred_fine['R4'], y_pred_fine['R5'])]
y_true_cat['Relevance'] = [r6 and r7 for r6, r7 in zip(y_true_fine['R6'], y_true_fine['R7'])]
y_pred_cat['Relevance'] = [r6 and r7 for r6, r7 in zip(y_pred_fine['R6'], y_pred_fine['R7'])]

results = {}

p, r, f1, _ = precision_recall_fscore_support(y_true_coh,
                                              y_pred_coh,
                                              average='macro',
                                              zero_division=1.0)
kappa = cohen_kappa_score(y_true_coh, y_pred_coh)
results['Coherence'] = {'Precison': p, 'Recall': r, 'F1': f1, 'Kappa': kappa}

for category_idx in ('Cohesion', 'Consistency', 'Relevance'):
    p, r, f1, _ = precision_recall_fscore_support(y_true_cat[category_idx],
                                                  y_pred_cat[category_idx],
                                                  average='macro',
                                                  zero_division=1.0)
    kappa = cohen_kappa_score(y_true_cat[category_idx], y_pred_cat[category_idx])
    results[category_idx] = {'Precison': p, 'Recall': r, 'F1': f1, 'Kappa': kappa}


for reason_idx in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'):
    p, r, f1, _ = precision_recall_fscore_support(y_true_fine[reason_idx],
                                                  y_pred_fine[reason_idx],
                                                  average='macro',
                                                  zero_division=1.0)
    kappa = cohen_kappa_score(y_true_fine[reason_idx], y_pred_fine[reason_idx])
    results[reason_idx] = {'Precison': p, 'Recall': r, 'F1': f1, 'Kappa': kappa}

results = pd.DataFrame.from_dict(results, orient='index')
print(results)