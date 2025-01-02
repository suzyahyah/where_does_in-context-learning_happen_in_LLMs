#!/usr/bin/python3
# Author: Suzanna Sia
# pylint: disable=C0303,C0103

### Standard imports

### Third Party imports
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer

### Custom imports
from src.datasets.bitext_dataset import get_fn_dataset
from src.datasets.prompt_dataset import PromptsDataset      

from src.utils import load_utils
from src.datasets.collate_fn import CollateFn


def build_model_tok(args_model):

    model, tokenizer = load_utils.get_models(args_model.model_size, 
                                             save_fol=args_model.save_fol,
                                             args_model=args_model)
    return model, tokenizer



def build_data_collator(args, tokenizer):
    if args.data_collator['type'] == "autoregressive_translation":
        data_collator = AutoRegressiveTranslationDC(tokenizer=tokenizer, mlm=False)
    elif args.data_collator['type'] == "autoregressive_lm":
        data_collator = AutoRegressiveLM_DC(tokenizer=tokenizer, mlm=False)

    else:
        raise NotImplementedError("not accounted for:", args.data_collator['type'])

    data_collator.my_collate_fn = CollateFn(tokenizer, cuda=False)
    return data_collator


def build_early_stop_callback(args):
    patience = args.early_stopping_cf.patience
    stopping_th = args.early_stopping_cf.threshold

    earlystop_cb = EarlyStoppingCallback(early_stopping_patience=patience,
                                         early_stopping_threshold=stopping_th) 
    return earlystop_cb


def build_datasets_for_prompt(args_data):

    # build a promptbank and testdataset 
    prompt_split = "train"
    test_split = "test"

    promptbank = get_fn_dataset(args_data.trainset, 
                                   prompt_split,
                                   args_data.direction, 
                                   data_path=args_data.train_data_fn)

    test_dataset = get_fn_dataset(args_data.testset, 
                             test_split,
                             args_data.direction, 
                             data_path=args_data.test_data_fn)
    return promptbank, test_dataset

def build_prompt_dataset(args, ds_promptbank, ds_test):
    prompt_ds = PromptsDataset(args.format, 
                               ds_promptbank, 
                               ds_test, 
                               seed=args.seed,
                               ntest=args.data.ntest)
    return prompt_ds
