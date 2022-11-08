#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export RUN_NAME=translate_text2img256_stable_diffusion_stochastic_custom
export SEED=42

python -m torch.distributed.launch --nproc_per_node 1 --master_port 1405 main.py \
    --seed $SEED \
    --cfg experiments/$RUN_NAME.cfg \
    --run_name $RUN_NAME$SEED \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --metric_for_best_model CLIPEnergy \
    --greater_is_better false \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 0 \
    --adafactor false \
    --learning_rate 1e-3 \
    --do_eval --output_dir output/$RUN_NAME$SEED \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 4 \
    --ddp_find_unused_parameters true \
    --verbose true > $RUN_NAME$SEED.log 2>&1 &

tail -f $RUN_NAME$SEED.log


# nohup python -m torch.distributed.launch --nproc_per_node 1 --master_port 1426 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$RUN_NAME$SEED --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --ddp_find_unused_parameters true --verbose true > $RUN_NAME$SEED.log 2>&1 &
