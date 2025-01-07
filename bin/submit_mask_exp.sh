#!/usr/bin/env bash
# Author: Suzanna Sia
#$ -l h=!r8n03*

cd $HOME/projects/where_does_in-context-learning_happen_in_LLMs

source "$1"
for key in "${!exp_args[@]}"; do
    echo $key: ${exp_args[$key]};
done

export TORCH_USE_CUDA_DSA=1 

PYTHONPATH=. python src/main.py \
    --data.direction ${exp_args['src-tgt']} \
    --format_cf configs/format/instr_${exp_args['instr']}.yaml \
    --model_cf configs/model/masks/mask_context_from_${exp_args['mask_conditions']}.yaml \
    --seed ${exp_args['seed']} \
    --data.trainset ${exp_args['trainset']} \
    --data.testset ${exp_args['testset']} \
    --data_cf configs/data/${exp_args['data_file']}.yaml \
    --model.model_size ${exp_args['model']} \
    --sample_prompts.nprompts ${exp_args['nprompts']} \
    --file_paths_cfg configs/file_paths/mask_context_from.yaml \
    --generator.batch_size 2
