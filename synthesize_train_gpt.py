import pandas as pd
import os
# from utils import *
from tqdm import tqdm
import random
from openai import OpenAI

import api_secrets
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)

def formulate_prompt(context, sentence):
    return [
        {"role": "system", "content": """
            You are an English teacher aiming to improve coherence in student writing. You are about to synthesize data for the coherence detection task. Concretely, for each data point, you will be given: a sentence S and a context C, which comprises all preceding sentences up to and immediately before sentence S in an essay written by an English second language learner. Then, you should follow the following steps to create a complete data point: \n1)  For sentence S and context C,  determine if sentence S is coherent with context C. You need to output 1 for [Coherence] if the sentence S is coherent when appended to the context C; otherwise, output 0; \n2) Then, if you output 1 in the previous step, output "Done" and finish; otherwise, move on to the following steps; \n3) You need to output 1 for [Reason 1] if the sentence S does not connect semantically with the context C; otherwise, output 0; \n4) You need to output 1 for [Reason 2] if the new sentence S discusses an entity that has not been introduced in C yet, or the new sentence S discusses an entity that is ambiguous in C; otherwise, output 0; \n5) You need to output 1 for [Reason 3] if the relation between sentence S and previous ones in C doesn't make sense due to a missing discourse marker; otherwise, output 0; \n6) You need to output 1 for [Reason 4] if the new sentence S contradicts or is inconsistent with previously presented information in C; otherwise, output 0; \n7) You need to output 1 for [Reason 5] if the new sentence S introduces information that is completely irrelevant to the context C; otherwise, output 0; \n8) You need to output 1 for [Reason 6] if the new sentence S introduces information that is either tangential or slightly irrelevant to the context C; otherwise, output 0; \n9) You need to output 1 for [Reason 7] if the comment (rheme/focus) of the sentence does not agree with the topic of the sentence; otherwise, output 0 \n10) [Rewrite] You should modify sentence S as minimally as possible to improve its coherence based on the following suggestions for each reason you might select above: \n- [Reason 1]: add reference words or repeated words or substitutions that can semantically connect sentence S to the context C; \n- [Reason 2]: link the newly introduced entity or ambiguous entity in S to the given context C \n- [Reason 3]: add or change a discourse marker that ties the sentence S with the given context C \n- [Reason 4]: align the newly introduced information with previously introduced information so that the new information in S does not contradict the context C \n- [Reason 5]: modify the sentence S so that it is relevant to the context C established by the writer \n- [Reason 6]: only output "DELETE" for deleting the sentence S \n- [Reason 7]: rewrite sentence S so that the comment of sentence S agrees with the topic of sentence S \n\nPlease disregard any incoherences in context C. You should output 1 for [Coherence] only if: \na) sentence S semantically connects to context C, and \nb) all entities discussed in the new sentence S have been introduced in C, and \nc) sentence S demonstrates reasonable discourse relation with previous ones, and  \nd) sentence S contains a meaning consistent with previously presented data in C, and \ne) sentence S contains a meaning relevant to previously presented data in C. \n\nHere are some examples:\nC: I believe that young people nowadays do not give enough time to helping their communities. \nS: This, i believe is caused by the environment we live in. \n- [Coherence]: 1 \n- Done \n\nC: Then, I wanna indicate that young people can study many things that are interesting or exciting things for young people. \nS: About students, they can learn various fields that students want to study. \n- [Coherence]: 0 \n- [Reason 1]: 1 \n- [Reason 2]: 0 \n- [Reason 3]: 1 \n- [Reason 4]: 0 \n- [Reason 5]: 0 \n- [Reason 6]: 0 \n- [Reason 7]: 0 \n- [Rewrite]: For example when they study, they can learn various fields that they want to study.\n\nC: There are three main reasons that my ideas support effectively, like action, study and knowledge. \nS: First of all, I wanna introduce young people's active points in comparison with older people. \n- [Coherence]: 0 \n- [Reason 1]: 0 \n- [Reason 2]: 0 \n- [Reason 3]: 0 \n- [Reason 4]: 1 \n- [Reason 5]: 0 \n- [Reason 6]: 0 \n- [Reason 7]: 0 \n- [Rewrite]: First of all, I wanna introduce young people's actions in comparison with older people's. \n\nC: These publicity agents use a lot of techniques to make the products look better, for example they use specialized software like photoshop to increase the size of the product or make it brighter, or maybe an artificial imitation of the product that does not necessarily have the same texture of look. \nS: Even though one can observe this situation mostly in food products.\n- [Coherence]: 0\n- [Reason 1]: 0\n- [Reason 2]: 0 \n- [Reason 3]: 0 \n- [Reason 4]: 0 \n- [Reason 5]: 0 \n- [Reason 6]: 1 \n- [Reason 7]: 0 \n- [Rewrite]: DELETE  \n\nC: I, however, think in terms of physical and mental factors young people are superior to older people. \nS: For example, in the case of sports young people can run and jump, and they can train their muscles that are used in each sport such as transitional sports or silence sports. \n- [Coherence]: 0 \n- [Reason 1]: 0 \n- [Reason 2]: 0 \n- [Reason 3]: 0 \n- [Reason 4]: 0 \n- [Reason 5]: 0 \n- [Reason 6]: 0 \n- [Reason 7]: 1 \n- [Rewrite]: For example, in the case of sports young people can run and jump, and they can train their muscles for sports more than older people can. \n\nNow, please generate:
    """},
        {"role": "user", "content": f"""
            C: {context} \nS: {sentence} \n
        """}
    ]

def detect_reason_rewrite(data_df):
    essay_ids = []
    topics = []
    contexts = []
    sentences = []
    labels = []
    reason1, reason2, reason3, reason4, reason5, reason6, reason7 = [], [], [], [], [], [], []
    rewrites = []
    for i in tqdm(range(len(data_df))):
        row = data_df.iloc[i]
        
        essay_id = row['essay_id']
        topic = row['topic']
        context = row['context']
        sentence = row['sentence']

        # formulate the prompt
        prompt = formulate_prompt(context, sentence)

        # initialize all items with empty string
        label = ''
        r1, r2, r3, r4, r5, r6, r7 = '', '', '', '', '', '', ''
        rewrite = ''

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                max_tokens=4096,
                temperature=0
            )
            # output = response['choices'][0]['message']['content']
            output = response.choices[0].message.content
            print(output)

            list_output = output.split('\n')
            label = list_output[0].split(': ')[1].strip()
            if label == '0':
                r1 = list_output[1].split(': ')[1].strip()
                r2 = list_output[2].split(': ')[1].strip()
                r3 = list_output[3].split(': ')[1].strip()
                r4 = list_output[4].split(': ')[1].strip()
                r5 = list_output[5].split(': ')[1].strip()
                r6 = list_output[6].split(': ')[1].strip()
                r7 = list_output[7].split(': ')[1].strip()
                rewrite = list_output[8].split(': ')[1].strip()

        except Exception as e:
            print(e)
            print('Error with index ', i)
            print('Context: ', context)
            print('Sentence: ', sentence)

        # append the results to the lists
        essay_ids.append(essay_id)
        topics.append(topic)
        contexts.append(context)
        sentences.append(sentence)
        labels.append(label)
        reason1.append(r1)
        reason2.append(r2)
        reason3.append(r3)
        reason4.append(r4)
        reason5.append(r5)
        reason6.append(r6)
        reason7.append(r7)
        rewrites.append(rewrite)

        # save the results to a csv file after every 1000 iterations
        if i % 1000 == 0:
            df = pd.DataFrame({
                'essay_id': essay_ids,
                'topic': topics,
                'context': contexts,
                'sentence': sentences,
                'label': labels,
                'R1': reason1,
                'R2': reason2,
                'R3': reason3,
                'R4': reason4,
                'R5': reason5,
                'R6': reason6,
                'R7': reason7,
                'Rewrite': rewrites
            })
            f_name = 'data/train/syn_train_{}.csv'.format(i)
            df.to_csv(f_name, index=False)

    return essay_ids, topics, contexts, sentences, labels, reason1, reason2, reason3, reason4, reason5, reason6, reason7, rewrites


data_inc_df = pd.read_csv('data/train/sample_759_inc.csv')
essay_ids, topics, contexts, sentences, labels, reason1, reason2, reason3, reason4, reason5, reason6, reason7, rewrites = detect_reason_rewrite(data_inc_df)

# save the results to a csv file
df_all = pd.DataFrame({
    'essay_id': essay_ids,
    'topic': topics,
    'context': contexts,
    'sentence': sentences,
    'label': labels,
    'R1': reason1,
    'R2': reason2,
    'R3': reason3,
    'R4': reason4,
    'R5': reason5,
    'R6': reason6,
    'R7': reason7,
    'Rewrite': rewrites
})
f_name = 'data/train/syn_train_all.csv'
df_all.to_csv(f_name, index=False)