#!/usr/bin/env bash
# Author: Suzanna Sia
SEEDS=(0 1 2) 
MODELS=(gptn125M) 
#{starcoder2-7B, llama7b, llama7b-chat, gptn2.7B, bloom3b)
task=(machine_translation) # {machine_translation,code_gen}

declare -A exp_args
exp_args['nprompts']=5

if [[ $task == "code_gen" ]]; then
    exp_args['trainset']="MBPP"
    exp_args['testset']="HEVAL"
    exp_args['src-tgt']="en-py"
    exp_args['data_file']="code_gen"

elif [[ $task == "machine_translation" ]]; then
    exp_args['trainset']="FLORES"
    exp_args['testset']="FLORES"
    exp_args['src-tgt']="fr-en"
    exp_args['data_file']="default"
fi

exps=("TTT,$task", "TTF,$task", "FTF,$task", "FTF,none_QA")

# write exp versions.
for seed in ${SEEDS[@]}; do
for model in ${MODELS[@]}; do 
for exp in ${exps[@]}; do
    IFS="," read -r mask instr <<< "$exp"
    exp_args['seed']="$seed"
    exp_args['model']="$model"
    exp_args['task']="$task"
    exp_args['mask_conditions']="$mask"
    exp_args['instr']="$instr"

    declare -p exp_args > "temp_bashargs"
    bash bin/submit_mask_exp.sh "temp_bashargs"
    rm "temp_bashargs"

done
done
done
