#!/bin/bash
# ./scripts/llava_instruct_1.5.yaml \
# /group/ossmodelzoo/zhenhliu/llava_instruct_1.5/coco118k_stage1.5_finetune_w_prompt.json
# /group/ossmodelzoo/zhenhliu/huggingface/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json
PYTHONPATH=./ torchrun --nproc_per_node 8 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /workspace/checkpoints/vlora-qwen2.5-3B-sft-sum-bs256/ \
    --version qwen_2 \
    --data_path ./scripts/single_image.yaml \
    --image_folder /group/ossmodelzoo/zhenhliu/Images_llava_instruct/ \
    --vision_tower /workspace/models/siglip-so400m-patch14-384 \
    --mm_projector_type vlora \
    --tune_vision_tower True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9  \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --group_by_modality_length True \
    --attn_implementation "flash_attention_2" \
    --bf16 True \
    --output_dir /workspace/checkpoints/vlora-qwen2.5-3B-train-sum-bs192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --vision_tower_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name vlora-qwen2.5-3b-train-sum-bs192 \
    --vlora_dim 512 \
    --vlora_depth 8 \
    --vlora_visual_dim 1152 \
    --vlora_pos_num 729 \
    --vlora_llm_dim 2048 \
    --vlora_llm_depth 36 \
    --vlora_rank 64 \
    --vlora_alpha 64 \
    --vlora_type qkvom \
    --vlora_type_group kv \
    --vlora_groups 8 \
    --weights_sep True \
    --skip_layers 4
