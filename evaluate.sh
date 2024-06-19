# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=20003 evaluate.py \
#     --model_name_or_path "/local/data/xuanming/models/output_vicuna_13b_train_combined_lr_1e_5/checkpoint-1149" \
#     --data_path data/dev/ProLex_v1.0_dev.csv \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
    # --bf16 True \
    # --output_dir /local/data/xuanming/models/output_vicuna_13b_trial_lr_1e_5 \
    # --num_train_epochs 10 \
    # --per_device_train_batch_size 1 \
    # --per_device_eval_batch_size 2 \
    # --gradient_accumulation_steps 1 \
    # --evaluation_strategy "no" \
    # --save_strategy "epoch" \
    # --save_steps 1200 \
    # --save_total_limit 10 \
    # --learning_rate 5e-5 \
    # --weight_decay 0. \
    # --warmup_steps 100 \
    # --lr_scheduler_type "linear" \
    # --logging_steps 1 \
    # --tf32 True \
    # --model_max_length 2048 \
    # --gradient_checkpointing True \
    # --lazy_preprocess False \

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

# python3 evaluate.py \
#     --model_name_or_path "gpt-4-32" \
#     --data_path data/test/test_final_cefr.csv \
#     --bf16 True \
#     --output_dir outputs/vicuna-7b-v1.5_trial \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1200 \
#     --save_total_limit 10 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --lazy_preprocess False \
