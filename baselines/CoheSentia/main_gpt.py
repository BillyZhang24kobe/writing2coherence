import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import openai
from openai import OpenAI
client = OpenAI()
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline
import torch
from tqdm import tqdm
import requests
import json

import signal
import datetime
import time
import gc
import argparse
from typing import Dict, List, Any
import csv
import random
# import matplotlib.pyplot as plt
from tqdm import tqdm

import datasets
from datasets import load_dataset, load_from_disk


SAVE_STEPS = 50

# model_engine = "ada" # You can choose any model from the list provided by OpenAI
# model = openai.Model(engine=model_engine)


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper


def get_final_answers(labels: List[str]) -> List[str]: 
    final_labels = []
    for label in labels:
        label = [l for l in label if l != '\n' and l != " "]
        label = "None" if len(label) == 0 else "".join(label)
        final_labels.append(label.lower())
    return final_labels


def write_answers(file_path: str, actual_labels: List[Any], generated_labels: List[Any]) -> None:
    final_preds, final_labels = [], []
    with open(file_path, "w") as writer:
        writer.write("predicted_label\treal_label\n")
        for pred, label in zip(generated_labels, actual_labels):
            # pred = [p for p in pred if p != '\n' and p != " "]
            # pred = "None" if len(pred) == 0 else "".join(pred)
            # final_preds.append(pred)
            
            # label = [l for l in label if l != '\n' and l != " "]
            # label = "None" if len(label) == 0 else "".join(label)
            # final_labels.append(label)
            
            writer.write(f"{pred}\t{label}\n")
    return 


def write_analysis(file_path: str, results: List[Dict[str, Any]], coherence_type: str):
    f1 = results[0]
    recall = results[2]
    precision = results[1]
    acc = results[3]
    with open(file_path, 'w') as fout1:
        tsv_writer = csv.writer(fout1, delimiter='\t')
        tsv_writer.writerow([f'coherence results analysis for data {coherence_type}:'])
        tsv_writer.writerow(['accuracy:'])
        tsv_writer.writerow([acc])

        tsv_writer.writerow(['f1:'])
        for eval_method, value in f1.items():
            tsv_writer.writerow([eval_method, value])

        tsv_writer.writerow(['recall:'])
        for eval_method, value in recall.items():
            tsv_writer.writerow([eval_method, value])

        tsv_writer.writerow(['precision:'])
        for eval_method, value in precision.items():
            tsv_writer.writerow([eval_method, value])


def get_from_all_classes(df: pd.DataFrame, label_name: str = 'label') -> pd.DataFrame:
    all_labels = list(set(df[label_name].tolist()))
    for i, l in enumerate(all_labels):
        new_df = df[df[label_name] == l].head(1)
        if i == 0:
            final_df = new_df
        else:
            final_df = pd.concat([final_df, new_df])
    return final_df


def get_data(path: str, type: str, debug_mode: bool = False) -> pd.DataFrame:
    path = path + type + '_incremental.csv'
    df = pd.read_csv(path)

    if debug_mode:
        df = get_from_all_classes(df, 'label')
    return df


def get_dataset(df: pd.DataFrame) -> List[Dict[str, str]]:
  final_dataset = []
  prompts, complitions = [], []
  title_prefix = 'Title: '
  text_prefix = 'Text: '
  question_prefix = 'What is the coherence score: '
  end_prefix = '\n###\n\n'
  # start_complition = ' The coherernce score is: '
  end_compiltion = '\n'

  for row_i, row in df.iterrows():
    text = row['text']
    title = row['title']
    score = row['label']

    prompt = title_prefix + title + '\n' + text_prefix + text + '\n' + question_prefix + '\n' + end_prefix
    complition = ' ' + str(score + 1) + end_compiltion
    # complition = start_complition + str(score + 1) + end_compiltion
    
    prompts.append(prompt)
    complitions.append(complition)
    final_dataset.append({"prompt": prompt, 'completion': complition})

  final_df = pd.DataFrame(list(zip(prompts, complitions)), columns=['prompt', 'completion'])
  return final_dataset


def load_data(path: str, data_type: str, task_type: str = 'score', debug_mode: bool = False) -> List[Dict[str, str]]:
    save_name = 'cached_{}_{}_{}_{}'.format(data_type, 'gpt', str(512), task_type)
    save_path = os.path.join(path, save_name)
    dataset = load_from_disk(save_path)
    df = dataset.to_pandas()
    if debug_mode:
        df = get_from_all_classes(df, 'completion')
    prompts = df['prompt'].tolist()
    complitions = df['completion'].tolist()
    strings = [{"prompt": prompt, 'completion': complition} for prompt, complition in zip(prompts, complitions)]
    return strings


# def convert_data_to_chat(strings: List[Dict[str, str]]) -> List[Dict[str, str]]:
#     role = "user"
#     strings = [{"role": role, "content": s['prompt'], "completion": s['completion']} for s in strings]
#     return strings


def prepare_data(path: str, data_type: str, task_type: str = 'score', load_mode: bool = True, debug_mode: bool = False, model_name: str = 'ada'):
    if load_mode or not task_type == 'score':
        strings = load_data(path, data_type, task_type, debug_mode)        
    else:
        df = get_data(path, data_type, debug_mode)
        strings = get_dataset(df)
    # path = os.path.join(path, model_name)
    # os.makedirs(path, exist_ok=True)
    final_file_name = os.path.join(path, data_type + '_data.jsonl')
    # if 'gpt-3.5' in model_name:
    #     strings = convert_data_to_chat(strings)

    with open(final_file_name, 'w') as outfile:
        for entry in strings:
            json.dump(entry, outfile)
            outfile.write('\n')
    return strings


# def check_data_in_gpt(dataset_path):
#     !openai tools fine_tunes.prepare_data -f dataset_path



def upload_file_to_openai(file_name: str) -> str:
    upload_response = openai.File.create(
        file=open(file_name, "rb"),
        purpose='fine-tune'
        )
    file_id = upload_response.id
    print(upload_response)
    return file_id


def signal_handler(sig, frame, job_id):
	status = openai.FineTune.retrieve(id=job_id).status
	print(f"Stream interrupted. Job is still {status}.")
	return


def save_result_file(job_id, output_file_path):
    os.makedirs(output_file_path, exist_ok=True)
    result_file_id = openai.FineTune.retrieve(job_id).result_files[0].id
    result_file_name = openai.FineTune.retrieve(job_id).result_files[0].filename
    # !openai api fine_tunes.results  -i job_id > file_path
    
    # Download the result file.
    print(f'Downloading result file: {result_file_id}')
    # Write the byte array returned by the File.download() method to 
    # a local file in the working directory.
    with open(output_file_path + result_file_name, "wb") as file:
        result = openai.File.download(id=result_file_id)
        file.write(result)
    results = pd.read_csv(output_file_path + result_file_name)
    return results[results['classification/accuracy'].notnull()]


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def fineune_with_backoff(**kwargs):
#     return openai.FineTune.create(**kwargs)

@retry_with_exponential_backoff
def fineune_with_backoff(**kwargs):
    return openai.FineTune.create(**kwargs)



def finetune(args, output_path: str) -> str:
    response = fineune_with_backoff(**args)
    # response = openai.FineTune.create(**args)
    job_id = response["id"]
    status = response["status"]

    print(f'Fine-tunning model with jobID: {job_id}.')
    print(f"Training Response: {response}")
    print(f"Training Status: {status}")
    fine_tune_events = openai.FineTune.list_events(id=job_id)

    print(f'Streaming events for the fine-tuning job: {job_id}')
    signal.signal(signal.SIGINT, signal_handler)
    events = openai.FineTune.stream_events(job_id)

    try:
        for event in events:
            print(f'{datetime.datetime.fromtimestamp(event["created_at"])} {event["message"]}')

    except Exception:
        print("Stream interrupted (client disconnected).")

    # Option 1 | if response.fine_tuned_model != null
    fine_tuned_model = response.fine_tuned_model

    if not fine_tuned_model:
        # Option 2 | if response.fine_tuned_model == null
        retrieve_response = openai.FineTune.retrieve(job_id)
        fine_tuned_model = retrieve_response.fine_tuned_model
        print(fine_tuned_model)
    else:
        print(fine_tuned_model)
    print(openai.FineTune.retrieve(id=job_id)["status"])

    status = openai.FineTune.retrieve(id=job_id)["status"]

    if status not in ["succeeded", "failed"]:
        print(f'Job not in terminal status: {status}. Waiting.')
        while status not in ["succeeded", "failed"]:
            time.sleep(2)
            status = openai.FineTune.retrieve(id=job_id)["status"]
            print(f'Status: {status}')
    else:
        print(f'Finetune job {job_id} finished with status: {status}')

    print('Checking other finetune jobs in the subscription.')
    result = openai.FineTune.list()
    print(f'Found {len(result.data)} finetune jobs.')

    if status == 'failed':
        print(f"finetuning failed with error {openai.FineTune.retrieve(id=job_id)}")
    fine_tuned_model = openai.FineTune.retrieve(id=job_id).fine_tuned_model
    print(f"the model name is {fine_tuned_model}")

    results = save_result_file(job_id, output_path)
    metric = results.tail(1)
    print(f"metric are {metric}")
    # results['classification/accuracy'].plot()
    return fine_tuned_model
    

def get_metrics(actual_labels: List[Any], generated_labels: List[Any]):
    f1, precision, recall = {}, {}, {}
    # Calculate the metrics
    accuracy = accuracy_score(actual_labels, generated_labels)

    f1['macro'] = f1_score(actual_labels, generated_labels, average='macro', zero_division=0)
    f1['weighted'] = f1_score(actual_labels, generated_labels, average='weighted', zero_division=0)
    f1['micro'] = f1_score(actual_labels, generated_labels, average='micro', zero_division=0)

    precision['macro'] = precision_score(actual_labels, generated_labels, average='macro', zero_division=0)
    precision['weighted'] = precision_score(actual_labels, generated_labels, average='weighted', zero_division=0)
    precision['micro'] = precision_score(actual_labels, generated_labels, average='micro', zero_division=0)

    recall['macro'] = recall_score(actual_labels, generated_labels, average='macro', zero_division=0)
    recall['weighted'] = recall_score(actual_labels, generated_labels, average='weighted', zero_division=0)
    recall['micro'] = recall_score(actual_labels, generated_labels, average='micro', zero_division=0)

    results = [f1, precision, recall, accuracy]
    return results


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def completion_with_backoff(**kwargs):
#     return openai.Completion.create(**kwargs)


# @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(6))
# def completion_chat_with_backoff(**kwargs):
#     return openai.ChatCompletion.create(**kwargs)


@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return client.completions.create(**kwargs)


@retry_with_exponential_backoff
def chat_completion_with_backoff(**kwargs):
    return client.chat.completions.create(model="gpt-4-turbo", messages=kwargs['messages'], temperature=0, max_tokens=1)
    return openai.ChatCompletion.create(**kwargs)


def inference(test_strings: List[Dict[str, str]], ft_model: str, output_path:str, coherence_type: str) -> None:
    num_examples = len(test_strings)
    new_prompts = [t['prompt'] for t in test_strings]
    actual_labels = [t['completion'] for t in test_strings]
    actual_labels = get_final_answers(actual_labels)
    all_pos_labels = list(set(actual_labels))
    checkpoint_path = output_path + 'checkpoint/'
    os.makedirs(checkpoint_path, exist_ok=True)
    answers_logs, answers = [], []
    for i, prompt in enumerate(tqdm(new_prompts)):
        # args = {
        #     "model":ft_model, 
        #     "prompt":prompt, 
        #     "max_tokens":1, 
        #     "temperature":0,
        #     "logprobs":len(all_pos_labels)
        #     }
        if 'gpt-3.5' in ft_model:
            message = [{"role": 'user', "content": prompt}]
            res = chat_completion_with_backoff(model=ft_model, messages=message, max_tokens=1, temperature=0)
            answers.append(res.choices[0].message.content)
        else:
            res = completion_with_backoff(model=ft_model, prompt=prompt, max_tokens=1, temperature=0, logprobs=len(all_pos_labels))
            # res = openai.Completion.create(**args)
            answers.append(res.choices[0].text)
            # answers_logs.append(res['choices'][0]['logprobs']['top_logprobs'][0])
        if (i % SAVE_STEPS) == 0 and i > 0:
            answers = get_final_answers(answers)
            checkpoint_save_path = checkpoint_path + coherence_type + '_checkpoint' + str(i) + '_results.txt'
            write_answers(checkpoint_save_path, actual_labels, answers)
    output_save_path = output_path + coherence_type + '_results.txt'
    write_answers(output_save_path, actual_labels, answers)
    results = get_metrics(actual_labels, answers)
    write_analysis(output_path + coherence_type + '_analysis_results.txt', results, coherence_type)
    return 


def get_parser():
    parser = argparse.ArgumentParser(description='Coherence')
    ## Required parameters 
    parser.add_argument("--output_dir", default="output/gpt/", type=str, help="where to save the model")
    parser.add_argument("--data_dir", default="data/", type=str, help="where is the data")
    parser.add_argument("--openai_key", default="", type=str, help="the openai key")

    parser.add_argument("--finetune", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    
    parser.add_argument("--mtl_mode", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--coherence_type", default="incremental", type=str, help="where the config is located")
    parser.add_argument("--classification_label", default="sent_binary", type=str, help="where the config is located")

    # Base Model 
    parser.add_argument("--model_name", default="gpt-3.5", type=str,  help="pretrained model")
    
    parser.add_argument("--load_mode", default=True, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--debug_mode", default=False, help="to debug or not", type=lambda x: (str(x).lower() == 'true'))
    # Data Params    
    parser.add_argument("--batch_size", default=16, type=int, help="the batch size not in debug mode")
    parser.add_argument("--num_epochs", default=5, type=int, help="the batch size not in debug mode")

    return parser.parse_args()


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu=0  

    args = get_parser() 
    # os.environ['OPENAI_API_KEY'] = args.openai_key
    num_epochs = 1 if args.debug_mode else args.num_epochs

    coherence_type = args.coherence_type if args.classification_label == 'score' else args.classification_label
    num_labels = 5 if args.classification_label == 'score' else 2
    if args.load_mode:
        data_dir = args.output_dir + coherence_type + '/' 
        if args.debug_mode:
            # data_dir = args.output_dir + 'debug/' + coherence_type + '/'
            output_dir_for_training = args.output_dir + '/debug/models/' + coherence_type + '/'
            file_output_name =  args.output_dir + 'debug/models/' + args.model_name + '/analysis/' 
        else:
            # data_dir = args.output_dir + coherence_type + '/' 
            output_dir_for_training = args.output_dir + '/models/' + coherence_type + '/'
            file_output_name =  args.output_dir + 'models/' + args.model_name + '/analysis/' 
    else:
        data_dir = args.data_dir
        if args.debug_mode:
            file_output_name =  args.output_dir + 'debug/models/' + args.model_name + '/analysis/' 
            output_dir_for_training = args.output_dir + '/debug/models/' + coherence_type + '/'
        else:
            file_output_name =  args.output_dir + 'models/' + args.model_name + '/analysis/' 
            output_dir_for_training = args.output_dir + '/models/' + coherence_type + '/'

    train_strings = prepare_data(data_dir, 'train', coherence_type, args.load_mode, args.debug_mode, args.model_name)
    dev_strings = prepare_data(data_dir, 'dev', coherence_type, args.load_mode, args.debug_mode, args.model_name)
    test_strings = prepare_data(data_dir, 'test', coherence_type, args.load_mode, args.debug_mode, args.model_name)

    print("Finish preparing data! \n")
    
    if args.finetune and not 'gpt-3.5' in args.model_name:
        training_file_id = data_dir + 'train_data.jsonl'
        dev_file_id = data_dir + 'dev_data.jsonl'
        training_file_id = upload_file_to_openai(training_file_id)
        dev_file_id = upload_file_to_openai(dev_file_id)

        model = args.model_name.split('-')[1]
        training_args = {
            "training_file": training_file_id,
            "validation_file": dev_file_id,
            "model": model,
            "n_epochs": num_epochs,
            "batch_size": args.batch_size,
            "learning_rate_multiplier": 0.3,
            "compute_classification_metrics" : True, 
            "classification_n_classes": num_labels
            }
        if num_labels == 2:
            training_args['classification_positive_class'] =  " no"
        print("Start Finetuning")
        ft_model = finetune(training_args, output_dir_for_training)
        print("Finish Finetuning")
    
    os.makedirs(file_output_name, exist_ok=True)
    print("Start Inference")
    if args.finetune and not 'gpt-3.5' in args.model_name:
        # file_output_name += coherence_type + '_analysis_results.txt'
        inference(test_strings, ft_model, file_output_name, coherence_type)
    else:
        file_output_name += 'only_predictions/'
        os.makedirs(file_output_name, exist_ok=True)
        # file_output_name += coherence_type + '_analysis_results.txt'

        final_string = []
        final_string.extend(train_strings)
        final_string.extend(dev_strings)
        final_string.extend(test_strings)
        inference(final_string, args.model_name, file_output_name, coherence_type)   
    print("Finish Inference")

