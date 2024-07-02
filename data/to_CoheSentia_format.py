import os
import pandas as pd

filename = 'test/cohesion_test.csv'  # CHANGE to file you would like to convert
target, split = 'cohesion', 'test'  # also change this to corresponding label and split

df = pd.read_csv(filename)
df.fillna(0, inplace=True)

# train = train.astype({'label': bool, 'R1': bool, 'R2': bool, 'R3': bool, 'R4': bool, 'R5': bool, 'R6': bool, 'R7': bool})
df = df.astype({'label': bool})

df = df.rename({'topic': 'title'}, axis=1)
df['coherence_per_sent'] = df['label']
df['cohesion_per_sent'] = df['label']
df['consistency_per_sent'] = df['label']
df['relevance_per_sent'] = df['label']

df['sents'] = df.apply(lambda row: [row['context'], row['sentence']], axis=1)
df['text'] = df.apply(lambda row: row['context'] + " " + row['sentence'], axis=1)
df['coherence_per_sent'] = df['coherence_per_sent'].map(lambda item: {'sent0': False, 'sent1': item})
df['cohesion_per_sent'] = df['cohesion_per_sent'].map(lambda item: {'sent0': False, 'sent1': item})
df['consistency_per_sent'] = df['consistency_per_sent'].map(lambda item: {'sent0': False, 'sent1': item})
df['relevance_per_sent'] = df['relevance_per_sent'].map(lambda item: {'sent0': False, 'sent1': item})

df.index.names = ['']
df = df[['title', 'text', 'sents', 'coherence_per_sent', 'cohesion_per_sent', 'consistency_per_sent', 'relevance_per_sent']]

os.makedirs('data_' + target, exist_ok=True)
df.to_csv(f'data_{target}/{split}_per_sent.csv')


