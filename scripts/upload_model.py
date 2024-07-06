from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("/local/data/xuanming/models/llama3_8b_instruct_train_rewriting_lr_1e_5_5epoch_1bsz_r_exp/checkpoint-4316")

tokenzier = AutoTokenizer.from_pretrained("/local/data/xuanming/models/llama3_8b_instruct_train_rewriting_lr_1e_5_5epoch_1bsz_r_exp/checkpoint-4316")

model.push_to_hub("Columbia-NLP/llama3-8b-instruct-rewriting-r-Decor")

tokenzier.push_to_hub("Columbia-NLP/llama3-8b-instruct-rewriting-r-Decor")