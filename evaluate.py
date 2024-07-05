from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import torch
import transformers

import pandas as pd
from utils import *
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from openai import OpenAI
import os
import random

import api_secrets
os.environ['OPENAI_API_KEY'] = api_secrets.openai_api_key

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=api_secrets.openai_org
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    task: str = field(
        default="detection", metadata={"help": "The task to train on. Select from detection, cohesion, consistency, relevance, other, and rewriting."}
    )
    lazy_preprocess: bool = False


class Evaluator(object):
    def __init__(self, model_args, model, eval_dataset, task, tokenizer=None, evaluate_all_metrics=False, print_results=False, with_gpt=False, rewriting_metric="acceptance_rate"):
        """ Initialize the Evaluator. 
        Args:
            args: TrainingArguments
            model: Pretrained model or model_name ('gpt-4', 'ChatGPT')
            tokenizer: Pretrained tokenizer
            eval_dataset: Path to the evaluation dataset
            metrics: 'hard' or 'soft'
            print_results: True or False
            aspect: 'acc' or 'prof'
            topk: top k candidates to be considered as substitutes
        """
        self.model_args = model_args
        self.model = model
        self.eval_dataset = eval_dataset
        self.task = task
        self.tokenizer = tokenizer
        self.evaluate_all_metrics = evaluate_all_metrics
        self.print_results = print_results
        self.with_gpt = with_gpt
        self.rewriting_metric = rewriting_metric

    def calculate_metrics(self, labels, preds):
        """ Calculate the precision, recall, and F1 score. 
        Args:
            labels: gold labels -> list of strings
            preds: predicted substitutes -> list of strings
        Returns:
            precision: precision score
            recall: recall score
            f1: F1 score
        """

        # Verify if labels and preds have the same length
        if len(labels) != len(preds):
            raise ValueError("The length of labels and preds must be the same")

        f1, precision, recall = {}, {}, {}
        # f1_real = {}
        
        # f1['macro'] = f1_score(labels, preds, average='macro')
        # f1['weighted'] = f1_score(labels, preds, average='weighted')
        # f1['micro'] = f1_score(labels, preds, average='micro')

        precision['macro'] = precision_score(labels, preds, average='macro')
        precision['weighted'] = precision_score(labels, preds, average='weighted')
        precision['micro'] = precision_score(labels, preds, average='micro')

        recall['macro'] = recall_score(labels, preds, average='macro')
        recall['weighted'] = recall_score(labels, preds, average='weighted')
        recall['micro'] = recall_score(labels, preds, average='micro')

        # f1['macro'] = 2 * (precision['macro'] * recall['macro']) / (precision['macro'] + recall['macro'])
        # f1['weighted'] = 2 * (precision['weighted'] * recall['weighted']) / (precision['weighted'] + recall['weighted'])
        # f1['micro'] = 2 * (precision['micro'] * recall['micro']) / (precision['micro'] + recall['micro'])

        f1['macro'] = f1_score(labels, preds, average='macro')
        f1['weighted'] = f1_score(labels, preds, average='weighted')
        f1['micro'] = f1_score(labels, preds, average='micro')

        results = [precision, recall, f1]
        return results
    
    def calculate_win_rate(self, eval_df, shuffle=True):
        selection2pos = {
            'm': 1,
            'M': 2
        }
        pos2selection = {
            1: 'm',
            2: 'M'
        }

        output = []
        selection = []
        pred_appearin_pos = []  # predictions positions (1 or 2)
        # pred_pos = []  # predictions positions (1 or 2)
        # gpt_selections = []  # GPT-4 selections ('m' or 'M')
        for i in tqdm(range(len(eval_df))):
            row = eval_df.iloc[i]
            context = row['context']
            sentence = row['sentence']
            pred = row['Predictions']
            ref = row['Rewrite']  # reference

            output_1 = pred  # first position
            output_2 = ref  # second position
            pred_pos = 1

            # randomly select between 1 and 2 for pred position
            if shuffle:
                pred_pos = selection2pos['m'] if random.random() > 0.5 else selection2pos['M']
                pred_appearin_pos.append(pred_pos)
                # pred_pos.append(pos)
                if pred_pos == 1:
                    output_1 = pred
                    output_2 = ref
                else:
                    output_1 = ref
                    output_2 = pred

            if 'rewriting-nr' in self.task:
                prompt = formulate_prompt_nr(context, sentence, output_1, output_2)
            elif 'rewriting-r' in self.task:
                r1, r2, r3, r4, r5, r6, r7 = row['R1'], row['R2'], row['R3'], row['R4'], row['R5'], row['R6'], row['R7']
                reasons = [r1, r2, r3, r4, r5, r6, r7]
                prompt = formulate_prompt_r(context, sentence, reasons, output_1, output_2)
            
            win = None
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=prompt,
                    max_tokens=2048,
                    temperature=0
                )
                # output = response['choices'][0]['message']['content']
                o = response.choices[0].message.content
                print("---------------------------")
                print("Index: ", i)
                print("Pred model id: ", pos2selection[pred_pos])
                print("Ref model id: ", pos2selection[3-pred_pos])
                print(o)
                judge_selection = o[-1]  # 'm' or 'M'
                selection.append(judge_selection)
                win = 1 if selection2pos[judge_selection] == pred_pos else 0  # 1 if the target model wins, 0 otherwise
            except Exception as e:
                print(e)
                print('Error with index ', i)
                print('Context: ', context)
                print('Sentence: ', sentence)
                print('Predicted: ', pred)
                print('Reference: ', ref)
                selection.append('')
            output.append(win)

        # save selection list to a file
        # pred_appearin_modelids = [pos2selection[p] for p in pred_appearin_pos]  # 'm' or 'M' for pred models
        if self.model_args.model_name_or_path.endswith(".csv"):
            model_path = self.model_args.model_name_or_path.split("/")[-1].split('.csv')[0]
        else:
            model_path = self.model_args.model_name_or_path.split("/")[-2]
        selection_pos = [selection2pos[s] for s in selection]
        with open("output/rewriting/{}_gpt_selections.txt".format(model_path), "w") as f:
            f.write("Pred model positions: " + str(pred_appearin_pos) + "\n")
            f.write("Selections pos: " + str(selection_pos) + "\n")
            f.write("--------------------------------------------------" + "\n")
            for i, row in eval_df.iterrows():
                context = row['context']
                sentence = row['sentence']
                pred = row['Predictions']
                ref = row['Rewrite']
                pred_position = pred_appearin_pos[i]

                f.write("Context: " + context + "\n")
                f.write("Sentence: " + sentence + "\n")
                f.write("Predicted: " + pred + "\n")
                f.write("Reference: " + ref + "\n")
                f.write("Prediction model id: " + pos2selection[pred_position] + "\n")
                f.write("Selection: " + selection[i] + "\n")
                f.write("--------------------------------------------------" + "\n")
                f.write("\n")

        # calculate the win rate and exclude None
        output = [o for o in output if o is not None]
        win_rate = sum(output) / len(output)

        return win_rate
    
    def calculate_acceptance_rate(self, eval_df):
        # calculate acceptance rate
        acceptance_rate = 0
        for i in tqdm(range(len(eval_df))):
            row = eval_df.iloc[i]
            context = row['context']
            # sentence = row['sentence']
            pred = row['Predictions']
            # ref = row['Rewrite']
            prompt = format_test_prompt_gpt_16shots_exp(context, pred)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=prompt,
                    max_tokens=2048,
                    temperature=0
                )
                o = response.choices[0].message.content
                print(o)
                o = o.split('\n')[0].strip()
                if int(o) == 1:
                    acceptance_rate += 1
                # acceptance_rate += int(o[-1] == 'm')
            except Exception as e:
                print(e)
                print('Error with index ', i)
                print('Context: ', context)
                # print('Sentence: ', sentence)
                print('Predicted: ', pred)
                # print('Reference: ', ref)

        acceptance_rate /= len(eval_df)
        return acceptance_rate
    
    def print_prediction_results(self, preds):
        """ Print the prediction results. """
        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        
        print("Printing prediction results...")
        print(preds)
        print("--------------------------------------------------")
        # store the predicted substitutes in a txt file
        with open("output/predictions.txt", "w") as f:
            for i, row in eval_df.iterrows():
                context = row['context']
                sentence = row['sentence']
                f.write("Context: " + context + "\n")
                f.write("Sentence: " + sentence + "\n")
                f.write("Gold answer: " + str(row['label']) + "\n")
                f.write("Predictions: " + str(preds[i]) + "\n")
                f.write("--------------------------------------------------" + "\n")
                f.write("\n")
    
    def evaluate_detection_metrics(self, labels, preds):
        return self.calculate_metrics(labels, preds)
    
    def evaluate_rewriting_gpt4(self, eval_df):
        if self.rewriting_metric == "acceptance_rate":
            return self.calculate_acceptance_rate(eval_df)
        elif self.rewriting_metric == "win_rate":
            return self.calculate_win_rate(eval_df, shuffle=True)
            

    def get_gold_labels(self):
        """ Get the gold labels from the evaluation dataset. """
        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        gold_labels = []

        for i in tqdm(range(len(eval_df))):
            row = eval_df.iloc[i]
            gold_labels.append(row['label'])
        
        return gold_labels

    def evaluate(self):
        """ Evaluate the model on the given dataset. """
        model_preds = []

        eval_df = pd.read_csv(self.eval_dataset, index_col=False)
        
        if self.model_args.model_name_or_path.endswith(".csv"):
        # self.model in ['human']:
            pred_df = pd.read_csv(self.model_args.model_name_or_path, index_col=False)
            model_preds = pred_df['Predictions']  # human
        elif 'gpt-4' in self.model_args.model_name_or_path:

            for i in tqdm(range(len(eval_df))):
                row = eval_df.iloc[i]

                context = row['context']
                sentence = row['sentence']

                if self.task == 'detection':
                    # system_input = format_test_prompt_gpt(context, sentence)
                    system_input = format_test_prompt_gpt_16shots_exp(context, sentence)
                elif self.task == 'cohesion':
                    system_input = format_test_prompt_cohesion_gpt_16shots_exp(context, sentence)
                elif self.task == 'consistency':
                    system_input = format_test_prompt_consistency_gpt(context, sentence)
                elif self.task == 'relevance':
                    system_input = format_test_prompt_relevance_gpt_16shots_exp(context, sentence)
                elif self.task == 'other':
                    system_input = format_test_prompt_other_gpt_16shots_exp(context, sentence)
                
                try:
                    response = client.chat.completions.create(
                        model=self.model_args.model_name_or_path,
                        messages=system_input,
                        max_tokens=2048,
                        temperature=0
                    )
                    # output = response['choices'][0]['message']['content']
                    output = response.choices[0].message.content
                    print(output)
                    pred = int(output.split('\n')[0].strip())
                except Exception as e:
                    print(e)
                    # print("Generated answer: ", generated_texts)
                    print("Error with index ", i)
                    print("Context: ", context)
                    print("Sentence: ", sentence)
                    pred = -1
                model_preds.append(pred)
        else:
            # eval mode
            self.model.eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # no gradient calculation
            with torch.no_grad():
                for i in tqdm(range(len(eval_df))):
                    row = eval_df.iloc[i]

                    context = row['context']
                    sentence = row['sentence']

                    if self.task == 'detection':
                        system_input = format_test_prompt(context, sentence)
                    # elif self.task in ['detection-llama3', 'cohesion-llama3', 'consistency-llama3', 'relevance-llama3', 'other-llama3']:
                    #     system_input = format_test_prompt_llama3(context, sentence)
                    elif self.task == 'cohesion':
                        system_input = format_test_prompt_cohesion(context, sentence)
                    elif self.task == 'consistency':
                        system_input = format_test_prompt_consistency(context, sentence)
                    elif self.task == 'relevance':
                        system_input = format_test_prompt_relevance(context, sentence)
                    elif self.task == 'other':
                        system_input = format_test_prompt_other(context, sentence)
                    elif self.task == 'rewriting-nr':
                        system_input = format_test_prompt_rewriting_nr(context, sentence)
                    elif self.task == 'rewriting-nr-llama3-instruct':
                        system_input = format_test_prompt_rewriting_nr_llama3_instruct(context, sentence)
                    elif self.task == 'rewriting-r':
                        r1, r2, r3, r4, r5, r6, r7 = row['R1'], row['R2'], row['R3'], row['R4'], row['R5'], row['R6'], row['R7']
                        reasons = [r1, r2, r3, r4, r5, r6, r7]
                        system_input = format_test_prompt_rewriting_r(context, sentence, reasons)
                    elif self.task == 'rewriting-r-llama3-instruct':
                        r1, r2, r3, r4, r5, r6, r7 = row['R1'], row['R2'], row['R3'], row['R4'], row['R5'], row['R6'], row['R7']
                        reasons = [r1, r2, r3, r4, r5, r6, r7]
                        system_input = format_test_prompt_rewriting_r_llama3_instruct(context, sentence, reasons)

                    if 'llama3' not in self.task:
                        input_ids = tokenizer.encode(system_input, return_tensors='pt', add_special_tokens=True)
                        input_ids = input_ids.cuda()

                        # Generate the candidates.
                        generated_ids = self.model.generate(
                            input_ids,
                            max_length=self.tokenizer.model_max_length,
                            temperature=0.2,
                            pad_token_id=self.tokenizer.pad_token_id)
                        
                        # Decode the candidates.
                        generated_texts = self.tokenizer.batch_decode(
                            generated_ids, skip_special_tokens=True)
                        
                        # print(generated_texts)
                        try:
                            if self.task in ['detection', 'cohesion', 'consistency', 'relevance', 'other']:
                                pred = int(generated_texts[0].split("Answer: ")[1].strip())
                            else:
                                pred = generated_texts[0].split("Answer: ")[1].strip()
                        except Exception as e:
                            print(e)
                            print("Generated answer: ", generated_texts)
                            pred = None
                        model_preds.append(pred)

                    else:
                        terminators = [
                            self.tokenizer.eos_token_id,
                            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
                        input_ids = self.tokenizer.apply_chat_template(
                            system_input,
                            add_generation_prompt=True,
                            return_tensors="pt"
                        ).to(model.device)
                        outputs = model.generate(
                            input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            pad_token_id=tokenizer.pad_token_id,
                            max_new_tokens=512,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                        )
                        generated_ids = outputs[0][input_ids.shape[-1]:]
                        generated_texts = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if 'rewriting' in self.task:
                            model_preds.append(generated_texts)
                        else:
                            model_preds.append(int(generated_texts))

        
        # print the results if print_results is True
        if self.print_results:
            self.print_prediction_results(model_preds)

        if 'rewriting' not in self.task:
        # self.task in ['detection', 'cohesion', 'consistency', 'relevance', 'other'] or self.task in ['detection-llama3', 'cohesion-llama3', 'consistency-llama3', 'relevance-llama3', 'other-llama3']:
            # calculate precision, recall, and F1 for the predictions
            gold_labels = self.get_gold_labels()
            assert len(gold_labels) == len(model_preds)
            print("Gold labels: ", gold_labels)
            print("Predictions: ", model_preds)
            results = self.evaluate_detection_metrics(gold_labels, model_preds)
            print("Precision: ", results[0])
            print("Recall: ", results[1])
            print("F1: ", results[2])
        else:
            if self.rewriting_metric == "acceptance_rate": # rewriting main metric
                eval_df['Predictions'] = model_preds
                if 'human' in self.model_args.model_name_or_path:
                    save_model_path = self.model_args.model_name_or_path.split("/")[-1].split('.csv')[0]
                else:
                    save_model_path = self.model_args.model_name_or_path.split("/")[-2]
                # check if file exists
                if not os.path.exists("output/rewriting/neg_448/{}_pred.csv".format(save_model_path)):
                    eval_df.to_csv("output/rewriting/neg_448/{}_pred.csv".format(save_model_path), index=False)
                print("Predictions saved to output")
                print("-------------------------------------------")
                print("Now, evaluating with GPT-4 ... ")
                acceptance_rate = self.evaluate_rewriting_gpt4(eval_df)
                print("Acceptance rate: ", acceptance_rate)
            else:
                # add a new column to the dataframe
                if not self.model_args.model_name_or_path.endswith(".csv"):
                    eval_df['Predictions'] = model_preds
                    save_model_path = self.model_args.model_name_or_path.split("/")[-2]
                    eval_df.to_csv("output/{}_pred.csv".format(save_model_path), index=False)
                    print("Predictions saved to output")
                    print("-------------------------------------------")
                    print("Now, evaluating with GPT-4 ... ")
                    win_rate = self.evaluate_rewriting_gpt4(eval_df)  # [1, ....] a list indicating if the target model wins (1) or not (0) compared to human reference -> win rate
                    print("Win rate: ", win_rate)
                else:
                    print("Now, evaluating with GPT-4 ... ")
                    win_rate = self.evaluate_rewriting_gpt4(eval_df)
                    print("Win rate: ", win_rate)


    def predict_single_turn(self, 
                             inputs: Tuple[str, str]):
        """ Predict substitutes given a Tuple of target word and sentence. """
        print("Predicting substitutes for target word: ", inputs[0])
        context, sentence = inputs
        system_input = format_test_prompt(context, sentence)

        # for input_text, target_text in zip(inputs, targets):

        input_ids = tokenizer.encode(system_input, return_tensors='pt', add_special_tokens=True)
        input_ids = input_ids.cuda()
        print("System input length: ", int(input_ids.ne(self.tokenizer.pad_token_id).sum()))

        # Generate the candidates.
        # eval mode
        self.model.eval()
        # no gradient calculation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_length=self.tokenizer.model_max_length,
                temperature=0.2)
            
        # Decode the candidates.
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)
        
        return generated_texts[0]
    


if __name__ == "__main__":

    # load model
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    if model_args.model_name_or_path.endswith(".csv"):
        # evaluate
        print("Evaluating predictions from ", model_args.model_name_or_path)
        # evaluator = Evaluator(model_args, model_args.model_name_or_path, data_args.data_path, task=data_args.task)
        evaluator = Evaluator(model_args, model_args.model_name_or_path, data_args.data_path, data_args.task, print_results=True)
        metrics = evaluator.evaluate()
    elif 'gpt-4' in model_args.model_name_or_path:
        print("Evaluating predictions from GPT-4 ...")
        evaluator = Evaluator(model_args, model_args.model_name_or_path, data_args.data_path, data_args.task, print_results=True, with_gpt=True)
        metrics = evaluator.evaluate()
    else:

        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            padding_side='left',
            use_fast=False,
            trust_remote_code=model_args.trust_remote_code,
        )

        device = torch.device('cuda:0')  
        model.to(device)
        model = model.bfloat16()

        # evaluate
        print("Evaluating...")
        print("Task: ", data_args.task)
        evaluator = Evaluator(model_args, model, data_args.data_path, data_args.task, tokenizer, print_results=True)
        metrics = evaluator.evaluate()


        # predict on single input
        # target_word = "obligatory"
        # sentence = "Even though it was an **obligatory** experience, I could take part in a community program"
        # inputs = (target_word, sentence)

        # evaluator = Evaluator(training_args, model, tokenizer, None)
        # generated_texts = evaluator.predict_single_turn(inputs)
        # print(generated_texts)