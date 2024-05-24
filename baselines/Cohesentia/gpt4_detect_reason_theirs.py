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
all_reasons = ["cohesive", "consistent", "relevant"]

def coherence_detection_prompt(context, sentence):
    return [
        {"role": "user", "content": f"Incoherence detection:\nPrevious Data: {context}\nNew Sentence: {sentence}\nTask: Is the new sentence coherent in regards to the previous data? give a yes or no answer\n\n###\n\n"}
    ]

@backoff.on_predicate(backoff.expo, lambda x: x == 'Error', max_tries=5)
def coherence_detection(context, sentence):
    prompt = coherence_detection_prompt(context, sentence)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = response.choices[0].message.content
    except Exception as e:
        print(e)
        output = 'Error'

    return output  # result or 'Error'

def coherence_reasoning_prompt(context, sentence, incoherent_reason):
    return [
        {"role": "user", "content": f"There are three dimensions of coherence: cohesion, consistency, and relevance.\nPrevious Data: {context}\nNew Sentence: {sentence}\nTask: Is the new sentence {incoherent_reason} in regards to the previous data? give a yes or no answer\n\n###\n\n"}
    ]

@backoff.on_predicate(backoff.expo, lambda x: x == 'Error', max_tries=5)
def coherence_reasoning(context, sentence, incoherent_reason):
    prompt = coherence_reasoning_prompt(context, sentence, incoherent_reason)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        output = response.choices[0].message.content
    except Exception as e:
        print(e)
        output = 'Error'

    
    return output  # result or 'Error'

with open(f'output/TestDataGPT3_theirs.json', 'w') as f:
    for entry in tqdm(test_processed):
        is_coherent = coherence_detection(entry['context'], entry['sentence'])
        entry['is_coherent_gpt'] = True if 'yes' in is_coherent.lower() else False
        entry['edit_gpt'] = []
        for reason in all_reasons:
            has_reason = coherence_reasoning(entry['context'], entry['sentence'], reason)
            if 'no' in has_reason.lower():
                entry['edit_gpt'].append({'reason': reason})
    json.dump(test_processed, f, ensure_ascii=False, indent=4)