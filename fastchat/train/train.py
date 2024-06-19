# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import pandas as pd
import ast
from torch.nn.utils.rnn import pad_sequence

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# system_prompt = """You are about to perform a lexical substitution task, considering the proficiency level of the substitute compared to the target word in a sentence. The task is to generate a set of candidate substitutes seperated by commas for a target word in a given sentence. The target word is highlighted in the sentence, encompassed by two double asterisks. The candidate substitutes should be: \n a) common collocations or expressions in actual English use, \n b) grammatically correct, \n c) have an equal or higher language proficiency level compared to the target word."""

def format_input_prompt(context, sentence):
    return """You are about to perform the task of coherence detection for the sentences written by second-language English learners. In this task, given a sentence S and a context C, you need to output 1 if S is coherent with C based on the following instructions; otherwise, output 0. You should output 1 only if: \na) sentence S semantically connects to context C, and \nb) all entities discussed in the new sentence S have been introduced in C, and \nc) the relation between sentence S and previous ones in C makes sense due to proper use of discourse markers, and \nd) the new sentence S does not contradict or is not inconsistent with previously presented information in C, and \ne) the new sentence S introduces information that is relevant to the context C established by the writer. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)

def format_input_prompt_llama3(context, sentence, label):
    return [
          {"role": "system", "content": """You are about to perform the task of coherence detection for the sentences written by second-language English learners. In this task, given a sentence S and a context C, you need to output 1 if S is coherent with C based on the following instructions; otherwise, output 0 . You should output 1 only if: \na) sentence S semantically connects to context C, and \nb) all entities discussed in the new sentence S have been introduced in C, and \nc) the relation between sentence S and previous ones in C makes sense due to proper use of discourse markers, and \nd) the new sentence S does not contradict or is not inconsistent with previously presented information in C, and \ne) the new sentence S introduces information that is relevant to the context C established by the writer. Now, please generate:"""},
          {"role": "user", "content": """C: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)},
          {"role": "assistant", "content": label}
        ]

def format_input_prompt_cohesion(context, sentence):
    return """You are about to perform the task of cohesion detection for the sentences written by second-language English learners. In this task, given a sentence S that is incoherent with a context C, you need to output 1 if S is incoherent with C because of a lack of cohesion based on the following instructions; otherwise, output 0. You should output 1 only if: \na) sentence S does not connect semantically with context C, or \nb) sentence S discusses an entity that has not been introduced in C yet, or sentence S discusses an entity that is ambiguous in C, or \nc) the relation between sentence S and previous ones in C doesn't make sense due to a missing discourse marker. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)


def format_input_prompt_consistency(context, sentence):
    return """You are about to perform the task of consistency detection for the sentences written by second-language English learners. In this task, given a sentence S that is incoherent with a context C, you need to output 1 if S contradicts previously presented information in C; otherwise, output 0. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)


def format_input_prompt_relevance(context, sentence):
    return """In this task, given a sentence S that is incoherent with a context C, you need to output 1 if S is incoherent with C because of a lack of relevance based on the following instructions; otherwise, output 0. You should output 1 only if: \na) sentence S introduces information that is completely irrelevant to context C, or \nb) sentence S introduces information that is either tangential or slightly irrelevant to context C. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)


def format_input_prompt_other(context, sentence):
    return """In this task, given a sentence S that is incoherent with a context C, you need to output 1 if S is incoherent with C because of the disagreement between the topic and the comment of sentence S; otherwise, output 0. Specifically, you should output 1 only if the comment of sentence S does not agree with the topic of the sentence itself. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)

# def format_input_prompt_rewriting_nr(context, sentence):
#     return """You are about to perform the task of sentence rewriting for the sentences written by second-language English learners. In this task, given a sentence S and a context C, you need to rewrite sentence S to make it more coherent with C. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)

def format_input_prompt_rewriting_nr(context, sentence):
    return """You are about to perform the task of sentence rewriting for the sentences written by second-language English learners. In this task, given a context C and a sentence S, where S is incoherent with C, you need to rewrite sentence S to make it coherent with C. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)

def format_input_prompt_rewriting_nr_llama3(context, sentence, response):
    return [
          {"role": "system", "content": """You are about to perform the task of sentence rewriting for the sentences written by second-language English learners. In this task, given a context C and a sentence S, where S is incoherent with C, you need to rewrite sentence S to make it coherent with C. Now, please generate:"""},
          {"role": "user", "content": """C: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)},
          {"role": "assistant", "content": response}]

def format_input_prompt_rewriting_r(context, sentence, reasons):
    r1, r2, r3, r4, r5, r6, r7 = reasons[0], reasons[1], reasons[2], reasons[3], reasons[4], reasons[5], reasons[6]
    reason_texts = ""
    if r1 == 1:
        reason_texts += "add reference words or repeated words/ideas or substitution that can semantically connect sentence S to context C. "
    if r2 == 1:
        reason_texts += "link the newly introduced entity or ambiguous entity in sentence S to context C. "
    if r3 == 1:
        reason_texts += "add or change a discourse marker that ties sentence S with context C. "
    if r4 == 1:
        reason_texts += "align the newly introduced information in sentence S with previously introduced information in context C so that the new information does not contradict the context. "
    if r5 == 1:
        reason_texts += "add information to sentence S that is relevant to the context C established by the writer. "
    if r6 == 1:
        reason_texts += "[DELETE]. "
    if r7 == 1:
        reason_texts += "rewrite sentence S so that the comment of sentence S agrees with the topic of the sentence itself. "


    return """You are about to perform the task of sentence rewriting for the sentences written by second-language English learners. In this task, given a context C and a sentence S, where S is incoherent with C, you need to rewrite sentence S to make it coherent with C according to the following instructions: {reason_texts}. Now, please generate: \nC: {context} \nS: {sentence} \n""".format(reason_texts=reason_texts, context=context, sentence=sentence)

def format_input_prompt_rewriting_r_llama3(context, sentence, reasons, response):
    r1, r2, r3, r4, r5, r6, r7 = reasons[0], reasons[1], reasons[2], reasons[3], reasons[4], reasons[5], reasons[6]
    reason_texts = ""
    if r1 == 1:
        reason_texts += "add reference words or repeated words/ideas or substitution that can semantically connect sentence S to context C. "
    if r2 == 1:
        reason_texts += "link the newly introduced entity or ambiguous entity in sentence S to context C. "
    if r3 == 1:
        reason_texts += "add or change a discourse marker that ties sentence S with context C. "
    if r4 == 1:
        reason_texts += "align the newly introduced information in sentence S with previously introduced information in context C so that the new information does not contradict the context. "
    if r5 == 1:
        reason_texts += "add information to sentence S that is relevant to the context C established by the writer. "
    if r6 == 1:
        reason_texts += "[DELETE]. "
    if r7 == 1:
        reason_texts += "rewrite sentence S so that the comment of sentence S agrees with the topic of the sentence itself. "

    return [
          {"role": "system", "content": """You are about to perform the task of sentence rewriting for the sentences written by second-language English learners. In this task, given a context C and a sentence S, where S is incoherent with C, you need to rewrite sentence S to make it coherent with C according to the following instructions: {reason_texts}. Now, please generate:""".format(reason_texts=reason_texts)},
          {"role": "user", "content": """C: {context} \nS: {sentence} \n""".format(context=context, sentence=sentence)},
          {"role": "assistant", "content": response}]


def format_target_prompt(label):
    return "Answer: {}</s>".format(label)


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


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    task: str,
) -> Dict:
    
    inputs = []
    targets = []
    llama3 = False
    if task in ["detection", "cohesion", "consistency", "relevance", "other"]:
        for i, source in enumerate(sources):
            context, sentence, label = source
            if task == 'detection':
                inputs.append(format_input_prompt(context, sentence))
            elif task == 'cohesion':
                inputs.append(format_input_prompt_cohesion(context, sentence))
            elif task == 'consistency':
                inputs.append(format_input_prompt_consistency(context, sentence))
            elif task == 'relevance':
                inputs.append(format_input_prompt_relevance(context, sentence))
            elif task == 'other':
                inputs.append(format_input_prompt_other(context, sentence))

            targets.append(format_target_prompt(label))
    elif task in ['detection-llama3', 'cohesion-llama3', 'consistency-llama3', 'relevance-llama3', 'other-llama3']:
        llama3 = True
        for i, source in enumerate(sources):
            context, sentence, label = source
            if task == 'detection-llama3':
                # if label == 1:
                #     label_yn = "Yes"
                # else:
                #     label_yn = "No"
                inputs.append(format_input_prompt_llama3(context, sentence, label))
            # elif task == 'cohesion-llama3':
            #     inputs.append(format_input_prompt_cohesion(context, sentence))
            # elif task == 'consistency-llama3':
            #     inputs.append(format_input_prompt_consistency(context, sentence))
            # elif task == 'relevance-llama3':
            #     inputs.append(format_input_prompt_relevance(context, sentence))
            # elif task == 'other-llama3':
            #     inputs.append(format_input_prompt_other(context, sentence))

    elif task == "rewriting-nr":
        for i, source in enumerate(sources):
            context, sentence, rewrite = source
            inputs.append(format_input_prompt_rewriting_nr(context, sentence))
            targets.append(format_target_prompt(rewrite))

    elif task == "rewriting-r":
        for i, source in enumerate(sources):
            context, sentence, r1, r2, r3, r4, r5, r6, r7, rewrite = source
            inputs.append(format_input_prompt_rewriting_r(context, sentence, [r1, r2, r3, r4, r5, r6, r7]))
            targets.append(format_target_prompt(rewrite))

    elif task == "rewriting-nr-llama3":
        llama3 = True
        for i, source in enumerate(sources):
            context, sentence, response = source
            inputs.append(format_input_prompt_rewriting_nr_llama3(context, sentence, response))
    elif task == "rewriting-r-llama3":
        llama3 = True
        for i, source in enumerate(sources):
            context, sentence, r1, r2, r3, r4, r5, r6, r7, response = source
            inputs.append(format_input_prompt_rewriting_r_llama3(context, sentence, [r1, r2, r3, r4, r5, r6, r7], response))

    # Apply prompt templates
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    if not llama3:
        for input_text, target_text in zip(inputs, targets):
            input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=True)[0]
            target_ids = tokenizer.encode(target_text, return_tensors='pt', add_special_tokens=False)[0]

            input_label_ids = torch.ones_like(input_ids) * -100
            target_label_ids = target_ids.clone()
            # mask out the target prompt words
            target_label_ids[:2] = -100

            final_input_ids = torch.cat([input_ids, target_ids], dim=-1)
            final_label_ids = torch.cat([input_label_ids, target_label_ids], dim=-1)

            all_input_ids.append(final_input_ids)
            all_labels.append(final_label_ids)
            all_attention_mask.append(torch.ones_like(final_input_ids))
    else:
        for input_texts in inputs:
            input_ids = tokenizer.apply_chat_template(
                input_texts,
                add_generation_prompt=False,
                return_tensors="pt"
            )[0]
            final_input_ids = input_ids
            final_label_ids = input_ids.clone()

            all_input_ids.append(final_input_ids)
            all_labels.append(final_label_ids)
            all_attention_mask.append(torch.ones_like(final_input_ids))
            
    try:
        final_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    except Exception as e:
        final_input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    print("input ids shape: ", final_input_ids.shape)
    final_labels = pad_sequence(all_labels, batch_first=True, padding_value=-100)
    final_attention_mask = pad_sequence(all_attention_mask, batch_first=True, padding_value=0).bool()

    return dict(
        input_ids=final_input_ids,
        labels=final_labels,
        attention_mask=final_attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, task: str):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        # if task in ["detection", "cohesion", "consistency", "relevance", "other"]:
        #     sources = [(context, sentence, label) for context, sentence, label in zip(raw_data['context'], raw_data['sentence'], raw_data['label'])]
        if "rewriting-nr" in task:
            # TODO
            sources = [(context, sentence, rewrite) for context, sentence, rewrite in zip(raw_data['context'], raw_data['sentence'], raw_data['Rewrite'])]
        elif "rewriting-r" in task:
            sources = [(context, sentence, r1, r2, r3, r4, r5, r6, r7, rewrite) for context, sentence, r1, r2, r3, r4, r5, r6, r7, rewrite in zip(raw_data['context'], raw_data['sentence'], raw_data['R1'], raw_data['R2'], raw_data['R3'], raw_data['R4'], raw_data['R5'], raw_data['R6'], raw_data['R7'], raw_data['Rewrite'])]
        else:
            sources = [(context, sentence, label) for context, sentence, label in zip(raw_data['context'], raw_data['sentence'], raw_data['label'])]

        # sources = [(t_word, t_sentence, subs) for t_word, t_sentence, subs in zip(raw_data['target_words'], raw_data['tagged_sentences'], raw_data['substitutes'].apply(lambda x : ast.literal_eval(x)))]
        data_dict = preprocess(sources, tokenizer, task)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    task = data_args.task
    train_csv = pd.read_csv(data_args.data_path, index_col=False)
    train_dataset = dataset_cls(train_csv, tokenizer=tokenizer, task=task)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, task=task)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = model.bfloat16()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()
