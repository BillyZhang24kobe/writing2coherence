import json
from collections import defaultdict
import random
prng = random.Random(45)
import backoff
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()

with open('data/TrainDataConverted.json') as f:
    train_processed = json.load(f)
with open('data/TestDataConverted.json') as f:
    test_processed = json.load(f)
all_data = train_processed + test_processed
all_reasons = ['R1: Sense - The sentence doesn’t make sense',
               'R2: Entity connection - The new sentence discusses an entity which has not been introduced yet',
               'R3: Discourse relation - The relation between this sentence and previous ones doesn’t make sense',
               'R4: Data consistency - The new sentence contains information inconsistent with previous presented data',
               'R5: World knowledge - The new sentence contains information inconsistent with your knowledge about the world',
               'R6: Data relevance - The new sentence is not relevant to previous data in the story',
               'R7: Title relevance - The new sentence is not relevant to the topic']

# # Identify reason
# coherence2example = defaultdict(list)
# reason2example = defaultdict(list)
# for example in train_processed:
#     coherence2example[example['is_coherent']].append(example)
#     for reason in example['edit']:
#         reason2example[reason['reason'][:2]].append(example)

# num_shot = 0
# detection_examples = [prng.choice(coherence2example[False]) for _ in range(num_shot // 2)] \
#                    + [prng.choice(coherence2example[True]) for _ in range(num_shot // 2)]
# reasoning_examples = [prng.choice(reason2example['R1']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R2']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R3']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R4']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R5']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R6']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R7']) for _ in range(num_shot // 8)] \
#                    + [prng.choice(reason2example['R8']) for _ in range(num_shot // 8)]

# Given reason, answer yes or no
coherence2example = defaultdict(list)
reason2example = {'R1': defaultdict(list),
                  'R2': defaultdict(list),
                  'R3': defaultdict(list),
                  'R4': defaultdict(list),
                  'R5': defaultdict(list),
                  'R6': defaultdict(list),
                  'R7': defaultdict(list)} # R1 -> True (has reason) / False (not this reason) -> example
for example in train_processed:
    coherence2example[example['is_coherent']].append(example)
    incoherent_reasons = set(reason['reason'][:2] for reason in example['edit'])
    for reason_idx in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'):
        if reason_idx in incoherent_reasons:
            reason2example[reason_idx][True].append(example)
        else:
            reason2example[reason_idx][False].append(example)

num_shot = 16
detection_examples = [prng.choice(coherence2example[False]) for _ in range(num_shot // 2)] \
                   + [prng.choice(coherence2example[True]) for _ in range(num_shot // 2)]
reasoning_examples = {'R1': {False: [prng.choice(reason2example['R1'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R1'][True]) for _ in range(num_shot // 2)]},
                      'R2': {False: [prng.choice(reason2example['R2'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R2'][True]) for _ in range(num_shot // 2)]},
                      'R3': {False: [prng.choice(reason2example['R3'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R3'][True]) for _ in range(num_shot // 2)]},
                      'R4': {False: [prng.choice(reason2example['R4'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R4'][True]) for _ in range(num_shot // 2)]},
                      'R5': {False: [prng.choice(reason2example['R5'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R5'][True]) for _ in range(num_shot // 2)]},
                      'R6': {False: [prng.choice(reason2example['R6'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R6'][True]) for _ in range(num_shot // 2)]},
                      'R7': {False: [prng.choice(reason2example['R7'][False]) for _ in range(num_shot // 2)], True: [prng.choice(reason2example['R7'][True]) for _ in range(num_shot // 2)]}}


def coherence_detection_prompt(context, sentence):
    context_examples = " \n \n ".join(f"C: {example['context']} \n S: {example['sentence']} \n -> {1 if example['is_coherent'] else 0}" for example in detection_examples)
    return [
        {"role": "system", "content": "You are an English teacher aiming to improve coherence in student writing. You are about to detect coherence in student writing. In this task, you will be given a context C and a sentence S. The context C comprises all sentences in a paragraph preceding sentence S. Please disregard any incoherences in the context C. You need to output 1 if the sentence S is coherent when appended to the context C; otherwise, output 0. \n \n "
         + (f"Here are some examples: \n {context_examples} \n \n " if num_shot > 0 else "")
         + "Now, start making predictions:"},
        {"role": "user", "content": f"C: {context} \n S: {sentence} \n ->"}
    ]

@backoff.on_predicate(backoff.expo, lambda x: x == 'Error', max_tries=5)
def coherence_detection(context, sentence):
    prompt = coherence_detection_prompt(context, sentence)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=prompt,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = int(response.choices[0].message.content)
    except Exception as e:
        print(e)
        output = 'Error'

    return output  # 0 or 1 or 'Error'

def coherence_reasoning_prompt(context, sentence, reason):
    reason_idx = reason[:2]
    incoherent_reason = reason.split(' - ')[1]

    # # Identify reason
    # context_examples = []
    # for example in reasoning_examples:
    #     reasons = ' \n '.join(reason['reason'] for reason in example['edit'])
    #     context_examples.append(f"C: {example['context']} \n S: {example['sentence']} \n -> {reasons}")
    # context_examples = " \n \n ".join(context_examples)

    # Given reason, answer yes or no
    context_examples = []
    for has_reason, examples in reasoning_examples[reason_idx].items():
        for example in examples:
            context_examples.append(f"C: {example['context']} \n S: {example['sentence']} \n -> {1 if has_reason else 0}")
    context_examples = " \n \n ".join(context_examples)

    return [
        {"role": "system", "content": f"You are an English teacher aiming to improve coherence in student writing. You are about to detect coherence in student writing. In this task, you will be given a context C and a sentence S. The context C comprises all sentences in a paragraph preceding sentence S. Please disregard any incoherences in the context C. You need to output 1 if {incoherent_reason}; otherwise, output 0. \n \n "
         + (f"Here are some examples: \n {context_examples} \n \n " if num_shot > 0 else "")
         + "Now, start making predictions:"},
        {"role": "user", "content": f"C: {context} \n S: {sentence} \n ->"}
    ]

@backoff.on_predicate(backoff.expo, lambda x: x == 'Error', max_tries=5)
def coherence_reasoning(context, sentence, incoherent_reason):
    prompt = coherence_reasoning_prompt(context, sentence, incoherent_reason)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=prompt,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = int(response.choices[0].message.content)
    except Exception as e:
        print(e)
        output = 'Error'

    
    return output  # 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8' or 'Error'

with open(f'output/AllDataGPT_{num_shot}shot.json', 'w') as f:
    for entry in tqdm(all_data):
        is_coherent = coherence_detection(entry['context'], entry['sentence'])
        entry['is_coherent_gpt'] = True if is_coherent == 1 else False
        entry['edit_gpt'] = []
        for reason in all_reasons:
            has_reason = coherence_reasoning(entry['context'], entry['sentence'], reason)
            if has_reason == 1:
                entry['edit_gpt'].append({'reason': reason})
    json.dump(all_data, f, ensure_ascii=False, indent=4)