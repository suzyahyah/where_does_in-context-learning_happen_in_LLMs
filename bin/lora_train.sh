#!/usr/bin/env bash
# Author: Suzanna Sia


dirc="fr-en"
model="Llama-3.1-8B"
for seed in 1 0; do
for trainsize in 400; do
  PYTHONPATH="." python src/lora_train.py --model.model_size $model --generator.batch_size 150 --data.direction $dirc --train_size $trainsize --seed $seed --start_layer 0 --end_layer -1
done
done
