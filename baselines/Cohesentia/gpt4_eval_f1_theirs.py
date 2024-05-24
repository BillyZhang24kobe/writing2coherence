import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

num_shot = 0

with open(f'output/TestDataGPT_0shot.json') as f:
    data = json.load(f)

y_true_coh, y_pred_coh = [], [] # True/False
# y_true_fine, y_pred_fine = [], [] # R1, R2, ..., R8
y_true_cat, y_pred_cat = [], [] # Cohesion, Consistency, Relevance, Other
reason2cat = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}
str_reason2cat = {"cohesive": 0, "consistent": 1, "relevant": 2}
for entry in data:
    if entry['is_coherent']:
        y_true_coh.append(True)
    else:
        y_true_coh.append(False)
    # edit_reasons = [0, 0, 0, 0, 0, 0, 0, 0]
    edit_cat = [0, 0, 0, 0]
    for reason in entry['edit']:
        reason_idx = int(reason['reason'][1])
        # edit_reasons[reason_idx-1] = 1
        edit_cat[reason2cat[reason_idx]] = 1
    # y_true_fine.append(edit_reasons)
    y_true_cat.append(edit_cat)
    if entry['is_coherent_gpt']:
        y_pred_coh.append(True)
    else:
        y_pred_coh.append(False)
    # edit_reasons = [0, 0, 0, 0, 0, 0, 0, 0]
    edit_cat = [0, 0, 0, 0]
    for reason in entry['edit_gpt']:
        # edit_reasons[reason_idx-1] = 1
        edit_cat[str_reason2cat[reason['reason']]] = 1
    # y_pred_fine.append(edit_reasons)
    y_pred_cat.append(edit_cat)

print('Coherence detection')
p, r, f1, _ = precision_recall_fscore_support(y_true_coh,
                                              y_pred_coh,
                                              average='binary',
                                              zero_division=1.0)
results = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Coherence'])
print(results)

print('\nCategorized reasons')
p, r, f1, _ = precision_recall_fscore_support(y_true_cat,
                                              y_pred_cat,
                                              average=None,
                                              zero_division=1.0)
results = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
                       index=['Cohesion', 'Consistency', 'Relevance', 'Other'])
print(results)

# print('\nFine-grained reasons')
# p, r, f1, _ = precision_recall_fscore_support(y_true_fine,
#                                               y_pred_fine,
#                                               average=None,
#                                               zero_division=1.0)
# results = pd.DataFrame({'Precison': p, 'Recall': r, 'F1': f1},
#                        index=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'])
# print(results)