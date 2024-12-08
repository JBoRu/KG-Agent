data_name=phi-2
size=2b

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

torchrun --nproc_per_node=8 --master_port=5999 train.py \
    --model_name_or_path /media/public/models/huggingface/microsoft/phi-2 \
    --data_path "./sft_mixture_train" \
    --bf16 True \
    --output_dir ./output/${data_name}-${size} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_z3_bf16.json \
    --tf32 True \
    --gradient_checkpointing True \
    --model_max_length 2048

# /usr/bin/python3.8 output/${data_name}-${size}/zero_to_fp32.py output/${data_name}-${size}/ output/${data_name}-${size}/pytorch_model.bin
# rm -r output/${data_name}-${size}/global*
