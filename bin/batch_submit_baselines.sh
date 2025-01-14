#!/usr/bin/env 
# Author: Suzanna Sia

SEEDS=(0)
#MODELS=(CodeLlama-7b-hf CodeLlama-7b-Instruct-hf)
MODELS=(starcoder2-3B) # run a small model to check the code runs
#MODELS=(Llama-3.1-8B) # run a small model to check the code runs
#MODELS=(starcoder2-15B llama7b llama7b-chat starcoder2-3B starcoder2-7B)

task=(code_gen) #{machine_translation,code_gen}
#task=(code_gen)

declare -A exp_args

exp_args['nprompts']=5
exp_args['task']="$task"
exp_args['instr']="$task"

if [[ $task == "code_gen" ]]; then
  exp_args['trainset']="MBPP"
  exp_args['testset']="HEVAL"
  exp_args['src-tgt']="en-py"
  exp_args['data_file']="default"

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
