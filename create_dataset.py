import pandas as pd


# input: essays is a list of dataframes
def create_annotation_dataset(essays):
    essay_ids = []
    topics = []
    contexts = []
    sentences = []
    labels = []
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    r5 = []
    r6 = []
    r7 = []
    rewrites = []

    for idx, essay in essays.iterrows():
    # for  in essays.iterrows():
        essay_id = essay['Filename']
        prompt_id = essay['Prompt']
        topic = prompt_dict[prompt_id]
        # if 'a.' in essay['corrected_essay'].values[0]:
        #     # replace a. with a
        #     essay['corrected_essay'] = essay['corrected_essay'].values[0].replace('a.', 'a ')
        essay_sents = [s.strip() for s in essay['corrected_essay'].split('.') if s.strip() != '']
        
        # initialize context with the first sentence
        context = essay_sents[0] + '.'
        for i in range(1, len(essay_sents)):
            essay_ids.append(essay_id)
            curr_sent = essay_sents[i] + '.'
            topics.append(topic)
            contexts.append(context)
            sentences.append(curr_sent)
            labels.append('')
            r1.append('')
            r2.append('')
            r3.append('')
            r4.append('')
            r5.append('')
            r6.append('')
            r7.append('')
            rewrites.append('')

            context += ' ' + curr_sent

    dataset = pd.DataFrame({
        'essay_id': essay_ids,
        'topic': topics,
        'context': contexts,
        'sentence': sentences,
        'label': labels,
        'R1': r1,
        'R2': r2,
        'R3': r3,
        'R4': r4,
        'R5': r5,
        'R6': r6,
        'R7': r7,
        'Rewrite': rewrites
    })
    
    return dataset

# load sampled raw medium-level essay data
train_raw = pd.read_csv('./data/raw/sample_759_raw.csv')

# create the incremental format of the essays
train_inc = create_annotation_dataset(train_raw)
train_inc.to_csv('./data/raw/sample_759_inc.csv', index=False)
