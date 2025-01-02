#!/usr/bin/env 
# Author: Suzanna Sia

SEEDS=(0 1 2)
MODELS=(gptn125M) # run a small model to check the code runs
#MODELS=(starcoder2-15B llama7b llama7b-chat starcoder2-3B starcoder2-7B)

task=(machine_translation) #{machine_translation,code_gen}

declare -A exp_args

exp_args['nprompts']=5
exp_args['task']="$task"
exp_args['instr']="$task"

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


for seed in ${SEEDS[@]}; do
for model in ${MODELS[@]}; do
  exp_args['seed']="$seed"
  exp_args['model']="$model"

  declare -p exp_args > "temp_bashargs"
  bash bin/submit_baselines.sh "temp_bashargs"
  rm "temp_bashargs"

done
done
