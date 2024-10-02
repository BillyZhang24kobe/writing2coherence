# 📔 DECOR: Improving Coherence in L2 English Writing with a Novel Benchmark for Incoherence Detection, Reasoning, and Rewriting
<img src="figures/decor_overview.png" width="100%">


This is the repository for DECOR, a novel benchmark that includes expert annotations for detecting incoherence in L2 English writing, identifying the underlying reasons, and rewriting the incoherent sentences. 

The figure above shows an example data point in DECOR, containing input context-sentence pairs, binary detection label, reason types for incoherence, and human rewrites.

In general, this repository offers:

1. The data format (CSV) for DECOR 📔
2. A supervised fine-tuning pipeline with task-specfic synthetic data
3. A standardized evaluation pipeline for all three tasks

## News
- [2024/09] 🎉 Our paper has been accepted to the EMNLP 2024 main conference!
- [2024/06] 🔥 We release the preprint of DECOR. Read our [paper](https://arxiv.org/abs/2406.19650) for more details!

## Table of Contents
- [Downloading the DECOR benchmark](#downloading-the-decor-benchmark)
- [Environment settings](#environment-settings)
- [Supervised Fine-tuning pipelines](#supervised-fine-tuning-pipelines)
- [Evaluating on DECOR](#evaluating-on-decor)
- [Citation](#citation)
- [Questions](#questions)

## Downloading the DECOR benchmark 📔
We release the dev and test data for each task in DECOR 📔. They can be downloaded from the following links in the table:


<table><thead>
  <tr>
    <th rowspan="2" style="text-align: center;"><br>Incoherence <br>Detection</th>
    <th colspan="4" style="text-align: center;">Incoherence reasoning</th>
    <th rowspan="2" style="text-align: center;"><br>Incoherence<br>Rewriting</th>
  </tr>
  <tr>
    <th style="text-align: center;">Cohesion</th>
    <th style="text-align: center;">Consistency</th>
    <th style="text-align: center;">Relevance</th>
    <th style="text-align: center;">Other</th>
  </tr></thead>
<tbody>
  <tr>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/binary_dev.csv">dev set</a>
    </td>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/cohesion_dev.csv">dev set</a>
    </td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/consistency_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/relevance_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/other_dev.csv">dev set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/dev/rewrite_541.csv">dev set</a></td>
  </tr>
  <tr>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/test_1355_clean.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/cohesion_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/consistency_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/relevance_test.csv">test set</a></td>
    <td>
    <a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/other_test.csv">test set</a></td>
    <td><a href="https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/test_rewrite_213_no_delete.csv">test set</a></td>
  </tr>
</tbody>
</table>

### DECOR data format

To construct DECOR, we start by creating context-sentence pairs from the essays sampled from [TOEFL-11 corpus](https://www.ets.org/research/policy_research_reports/publications/report/2013/jrkv.html). Each data point of DECOR consists of the following columns:

- `essay_id`: the id of the original essay from TOEFL-11 dataset.
- `topic`: the topic prompt as plain text for the sampled essay. Note that in total there are 8 prompts from the dataset.
- `context`: the context sentences as plain text, which comprises all preceding sentences up to and immediately before the current sentence in the essay.
- `sentence`: the current sentence as plain text
- `label`: a binary label (1 or 0) indicating (a) if the current sentence is incoherent with the context for the task of incoherence detection, or (b) if the incoherence is caused by any of the reasons (i.e. cohesion, consistency, relevance, and other) for the task of incoherence reasoning.
- `R1`: a binary label (1 or 0) indicating if the incoherence is caused by _semantic connection_.
- `R2`: a binary label (1 or 0) indicating if the incherence is caused by _entity reference_.
- `R3`: a binary label (1 or 0) indicating if the incherence is caused by _discourse relation_.
- `R4`: a binary label (1 or 0) indicating if the incherence is caused by _consistency_.
- `R5`: a binary label (1 or 0) indicating if the incherence is caused by _contextual relevance_.
- `R6`: a binary label (1 or 0) indicating if the incherence is caused by _tangential relevance_.
- `R7`: a binary label (1 or 0) indicating if the incherence is caused by _other_ reasons that are not specified above.
- `Rewrite`: The rewrite of the incoherent sentence so that it is now coherent with the context.

Note that for coherent context-sentence pairs, the reason types and rewrites are marked empty.

## Environment settings
### Conda environment installation
```
git clone https://github.com/BillyZhang24kobe/writing2coherence.git
cd writing2coherence
conda env create -f environment.yml
conda activate w2c
```

## Supervised Fine-tuning pipelines
We provide scripts to synthesize task-specific training data with GPT-4, and then fine-tune `Llama2-7B` and `Llama3-8B-Instruct` on the synthetic data.

### Synthesize task-specific training data
#### Create source dataset for prompting
We start by creating the source dataset in an incremental format. Running `python3 create_dataset.py` takes in the raw sampled essays (i.e. `sample_759_raw.csv`) as the input and convert it into the incremental format (i.e. `sample_759_inc.csv`).

#### Generate annotation with GPT-4
Generate the synthetic data with GPT-4 using the following command. Note that you should create an `api_secrets.py` file in the root directory of the project, and input your OpenAI API credentials to the file before running the script.
```
python3 synthesize_gpt.py
```
Specifically, the script takes as an input the source incremental data from `data/raw/sample_759_inc.csv`. The output is stored in `data/train/syn_train.csv`. All synthetic training data for all tasks is stored in the `data/train/decor` folder.

### Supervised Fine-tuning recipes
We employ the following scripts to fine-tune the `Llama2-7B` and `Llama3-8B-Instruct` models. We provide the configuration details to help you replicate our experiment results. We specifically build our pipeline based on [FastChat](https://github.com/lm-sys/FastChat). Please fill in `DEVICE`, `MODEL_PATH`, `DATA_PATH`, `OUTPUT_MODEL_PATH`, and `TASK_NAME` accordingly. Please specify `TASK_NAME` from the following list: `detection`, `cohesion`, `consistency`, `relevance`, `other`, `rewriting-r`, and `rewriting-r-llama3`.
```
CUDA_VISIBLE_DEVICES={DEVICE} python fastchat/train/train_mem.py \
    --model_name_or_path {MODEL_PATH} \
    --data_path {DATA_PATH} \
    --bf16 True \
    --output_dir {OUTPUT_MODEL_PATH} \
    --task {TASK_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
```

### Model Weights
We release the model weights for the task of incoherence rewriting.
#### Llama2-7B Weights
| Training Condition | Description | Hugging Face Repo |
| ---  | --- | --- |
| w/ reason   | Our Llama2-7B model finetuned on our task-specific synthetic data with the reasons for incoherence.  | [Columbia-NLP/llama2-7b-rewriting-r-Decor](https://huggingface.co/Columbia-NLP/llama2-7b-rewriting-r-Decor)   |
| w/o reason   | Our Llama2-7B model finetuned on our task-specific synthetic data without the reasons for incoherence. | [Columbia-NLP/llama2-7b-rewriting-nr-Decor](https://huggingface.co/Columbia-NLP/llama2-7b-rewriting-nr-Decor)   |

#### Llama3-8B-Instruct Weights
| Training Condition | Description | Hugging Face Repo |
| ---  | --- | --- |
| w/ reason   | Our Llama3-8B-Instruct model finetuned on our task-specific synthetic data with the reasons for incoherence.  | [Columbia-NLP/llama3-8b-instruct-rewriting-r-Decor](https://huggingface.co/Columbia-NLP/llama3-8b-instruct-rewriting-r-Decor)   |
| w/o reason   | Our Llama3-8B-Instruct finetuned on our task-specific synthetic data without the reasons for incoherence. | [Columbia-NLP/llama3-8b-instruct-rewriting-nr-Decor](https://huggingface.co/Columbia-NLP/llama3-8b-instruct-rewriting-nr-Decor)   |

## Evaluating on DECOR
We provide the evaluation pipeline for all three tasks proposed in DECOR. The following code snippet demonstrates how to plug in a rewriting model fine-tuned based on `Llama3-8B-Instruct` to generate rewrites for the given incoherent context-sentence pair. Feel free to replace the model checkpoints with your own checkpoints.

```py
import torch
import transformers
from utils import *

# load model and tokenizer
config = transformers.AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
model = transformers.AutoModelForCausalLM.from_pretrained(
                                                 MODEL_NAME_OR_PATH, config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(
                           MODEL_NAME_OR_PATH, padding_side='left', use_fast=False)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 
model.to(device)
model = model.bfloat16()

# An incoherent context-sentence pair
context = "In general, many people like young people enjoy life more than older people do. I agree with this statement in terms of young men's advantages. There are three main reasons that my ideas support effectively, like action, study and knowledge."
sentence = "First of all, I wanna introduce young people's active points in comparison with older people."

# Format input prompt
system_input = format_test_prompt_rewriting_nr_llama3_instruct(context, sentence)
input_ids = tokenizer.apply_chat_template(system_input,
                                          add_generation_prompt=True,
                                          return_tensors="pt").to(model.device)

# terminator configuration
terminators = [tokenizer.eos_token_id, 
               tokenizer.convert_tokens_to_ids("<|eot_id|>")]


# generate outputs
outputs = model.generate(input_ids,
                         attention_mask=torch.ones_like(input_ids),
                         pad_token_id=tokenizer.pad_token_id,
                         max_new_tokens=512,
                         eos_token_id=terminators,
                         do_sample=True,
                         temperature=0.2,
                         top_p=0.9)
generated_ids = outputs[0][input_ids.shape[-1]:]
generated_texts = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
```

### Incoherence Detection
To evaluate model performance in detecting incoherence (i.e. Yes or No for binary detection) on the [test set](https://github.com/BillyZhang24kobe/writing2coherence/blob/main/data/test/test_1355_clean.csv), run the following command: (remember to replace MODEL_NAME_OR_PATH with your own model path)
```
python3 evaluate.py \
     --model_name_or_path {MODEL_NAME_OR_PATH} \
     --data_path data/test/test_1355_clean.csv \
     --task "detection" 
```
### Incoherence Reasoning
We divide the overall incoherence reasoning task into four distinct sub-tasks, each targeting a different cause. To evaluate model's ability in determining whether the incoherence stems from a specific cause, run the following command with your own model checkpoints and specify SUB_TASK from the list (i.e. `cohesion`, `consistency`, `relevance`, and `other`):
```
python3 evaluate.py \
     --model_name_or_path {MODEL_NAME_OR_PATH} \
     --data_path data/test/{SUB_TASK}_test.csv \
     --task SUB_TASK
```
### Incoherence Rewriting
We evaluate the performance of incoherence rewriting with GPT-4 using two automaic metrics: `acceptance_rate` and `win_rate`. The default metric is set to `acceptance_rate`, but you can change it in the Evaluator constructor at line 54 of `evaluate.py`. Run the following command to evaluate rewrites generated by your models.
```
python3 evaluate.py \
     --model_name_or_path {MODEL_NAME_OR_PATH} \
     --data_path data/test/test_rewrite_213_no_delete.csv \
     --task "rewriting-r" 
``` 

## Citation
We highly appriciate your interests in our work. If you find DECOR 📔 helpful, please consider citing our paper in your work:

Xuanming Zhang, Anthony Diaz, Zixun Chen, Qingyang Wu, Kun Qian, Erik Voss, Zhou Yu. [DECOR: Improving Coherence in L2 English Writing with a Novel
Benchmark for Incoherence Detection, Reasoning, and Rewriting](https://arxiv.org/abs/2406.19650)

```
@article{zhang2024decor,
  title={DECOR: Improving Coherence in L2 English Writing with a Novel
Benchmark for Incoherence Detection, Reasoning, and Rewriting},
  author={Zhang, Xuanming, and Diaz, Anthony, and Chen, Zixun, and Wu, Qingyang, and Qian Kun, and Voss, Erik, and Yu, Zhou},
  journal={arXiv preprint arXiv:2406.19650},
  year={2024}
}
```

## Questions
Please reach out to us at billyzhang@cs.columbia.edu if you have any questions in using our benchmark. If you find an issue in either the source code or dataset, please feel free to create a pull request and make contribution to the benchmark!
