# CUDA_VISIBLE_DEVICES=4 python3 evaluate.py \
#     --model_name_or_path "/local/data/xuanming/models/llama3_8b_instruct_train_detection_lr_1e_5_5epoch_1bsz_1_0/checkpoint-40560" \
#     --data_path data/test/test_1355_clean.csv \
#     --task "detection-llama3" \

# python3 evaluate.py \
#     --model_name_or_path "gpt-4" \
#     --data_path data/test/other_test.csv \
#     --task "other" \

# python3 evaluate.py \
#     --model_name_or_path "output/rewriting/win_rate_213_human_as_ref/meta-llama_pred_clean.csv" \
#     --data_path output/rewriting/win_rate_213_human_as_ref/meta-llama_pred_clean.csv \
#     --task "rewriting-r" \

# python3 evaluate.py \
#     --model_name_or_path "/home/billyzhang/writing_coherence/writing2coherence/output/rewriting/neg_448/llama2_7b_train_rewriting_lr_1e_5_5epoch_1bsz_nr_exp_pred.csv" \
#     --data_path data/test/test_neg_448.csv \
#     --task "rewriting-r" \

CUDA_VISIBLE_DEVICES=5 python3 evaluate.py \
    --model_name_or_path "/local/data/xuanming/models/llama2_7b_train_relevance_lr_2e_5_5epoch_1bsz_cs/checkpoint-9988" \
    --data_path data/test/relevance_test.csv \
    --task "relevance" \

# CUDA_VISIBLE_DEVICES=3 python3 evaluate.py \
#     --model_name_or_path "/local/data/xuanming/models/llama3_8b_instruct_train_rewriting_lr_1e_5_5epoch_1bsz_r_exp/checkpoint-4316" \
#     --data_path data/test/test_rewrite_213_no_delete.csv \
#     --task "rewriting-r-llama3-instruct" \
