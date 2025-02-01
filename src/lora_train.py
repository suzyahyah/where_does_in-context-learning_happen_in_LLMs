#!/usr/bin/python3
# Author: Suzanna Sia

# Standard Imports
import argparse
from functools import partial
from typing import List, Dict

# Third party imports
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from transformers import PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


# Local imports
from src.utils import build, io_utils, utils

def collate_fn(batch: List[Dict], tokenizer: PreTrainedTokenizer, test=False, max_length: int = 512):
    """
    Collate function for preparing data for causal language modeling using HuggingFace Trainer.
    
    Args:
        batch: List of dictionaries containing 'source' and 'target' text pairs
        tokenizer: HuggingFace tokenizer for encoding the text
        max_length: Maximum sequence length for truncation
        
    Returns:
        Dictionary containing input_ids, attention_mask, and labels tensors
    """
    # Combine source and target with a separator token
    if test:
        combined_texts = [f"Q:{item['source']}\nA:" for item in batch]

    else:
        combined_texts = [
            f"Q:{item['source']}\nA:{item['target']}{tokenizer.eos_token}"
            for item in batch
        ]
        
    # Tokenize all texts in the batch
    encodings = tokenizer(
        combined_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create the labels tensor (same as input_ids for causal LM)
    if test:
        labels = [item['target'] for item in batch]

    else:
        labels = encodings.input_ids.clone()
    
    # Mask padding tokens with -100 so they're ignored in the loss
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "labels": labels
    }

def lora_specific_layers(model_name, layer=0):
    if "gpt" in model_name:
        target_mod = f"transformer.h.{layer}.attn.attention.out_proj"
    elif "llama" in model_name.lower():
        target_mod = f"model.layers.{layer}.self_attn.o_proj"
    target_modules = [target_mod]
    lora_config = LoraConfig(
        r=8,                      # LoRA attention dimension
        lora_alpha=32,           # Alpha scaling
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config

def get_training_args():

    training_args = TrainingArguments(
            output_dir="models",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            logging_dir="./logs",
            eval_steps=100,
            evaluation_strategy='steps',
            save_strategy='steps',
            learning_rate=1e-4,
            load_best_model_at_end=True,
            seed=args.seed)
    return training_args

def main(args, cfp):
    model, tok = build.build_model_tok(args.model)
    train_ds, eval_ds, test_ds = build.build_datasets_for_train(args.data, args.train_size)
    if not args.run_without_train:
        lora_config = lora_specific_layers(args.model.model_size, args.model.train_layer)

        model = get_peft_model(model, lora_config)
        collate_w_tokenizer = partial(collate_fn, tokenizer=tok)
        training_args = get_training_args()
        early_stop = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.01)]

        trainer = Trainer(model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tok,
                data_collator=collate_w_tokenizer,
                callbacks=early_stop,
                )
        trainer.train()
        model.save_pretrained(f'models/lora/{args.model.model_size}/layer-{args.model.train_layer}/{args.data.direction}-{args.train_size}')

    test_collate = partial(collate_fn, tokenizer=tok, test=True)
    dataloader = DataLoader(test_ds, collate_fn=test_collate, batch_size=args.generator.batch_size)
    all_gen_text = utils.generation(args.hf_generator_configs,
                     args.format,
                     args.model,
                     model,
                     tok,
                     dataloader)
    io_utils.save_and_eval(cfp, args, all_gen_text)
					 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--format_cf', default='configs/format/instr_none_QA.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--train_size', default=800, type=int)
    argparser.add_argument('--file_paths_cfg', default='configs/file_paths/lora.yaml')
    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--do_baseline', action='store_true')
    argparser.add_argument('--eval_only', action='store_true')
    argparser.add_argument('--run_without_train', action='store_true')
    argparser.add_argument('--start_layer', default=0, type=int)
    argparser.add_argument('--end_layer', default=-1, type=int)

    args, uk_args = argparser.parse_known_args()
    args = io_utils.merge_omegaconf_w_argparse(args, uk_args)
    cfp = OmegaConf.load(args.file_paths_cfg)
    if args.end_layer != -1:
        end_layer = args.end_layer
    else:
        end_layer = 32

    for layer in range(args.start_layer, end_layer):
        args.model.train_layer = layer
        main(args, cfp)
