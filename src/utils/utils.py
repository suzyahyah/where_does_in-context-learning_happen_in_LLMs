#!/usr/bin/python3
# Author: Suzanna Sia

# Standard imports
from tqdm import tqdm
import pandas as pd
import torch
from src.utils import nvidia_utils
from transformers import StoppingCriteria, StoppingCriteriaList

def generation(hf_generator_cf, 
               format_cf, 
               model_cf,
               model, 
               tokenizer, 
               dataloader):

    all_gen_text = []
#    stopping_criteria = StoppingCriteriaList([StopOnTokens(tokenizer, format_cf, model_cf)])
    stopping_criteria = None

    running_id = 0
    if model.device.type == "cpu":
        model = model.cuda()
    for j, batch in enumerate(tqdm(dataloader)):
        build_causal_mask_per_batch(model_cf, model, batch)
        #prompt_len = max(batch['prompt_len'])
        #if "FLORES" in dataloader.dataset.ds_promptbank.name:
        #    max_new_tokens = ((prompt_len - batch['instructions_len'][0]) // dataloader.dataset.nprompts)  * 0.5
        #    print("Max new tokens for translation:", max_new_tokens)
        #else:
        #    max_new_tokens = ((prompt_len - batch['instructions_len'][0]) // dataloader.dataset.nprompts) * 2
        max_new_tokens = 50
        if batch['input_ids'].device.type == 'cpu':
            batch['input_ids'] = batch['input_ids'].cuda()
            batch['attention_mask'] = batch['attention_mask'].cuda()


        with torch.no_grad():
            outputs = model.generate(batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     pad_token_id=tokenizer.pad_token_id,
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     stopping_criteria=stopping_criteria,
                                     max_new_tokens=max_new_tokens,
                                     **hf_generator_cf)

        gen_ids = outputs.sequences
        # later versions of huggingface dont need to find start_ix
        start_ix = batch['input_ids'].shape[1]
        gen_text = tokenizer.batch_decode(gen_ids[:, start_ix:])
        gen_text = clean_up(gen_text, format_cf, tokenizer)

        if j == 0:
            # vibe check
            print(tokenizer.decode(batch['input_ids'][0]))
            print("\n====")
            print("raw gen:", tokenizer.batch_decode(gen_ids[:, start_ix:])[0])
            print("cleanup:", gen_text[0])
            out = nvidia_utils.print_gpu_utilization()
            print(out)

        for i in range(len(gen_text)):
            #all_gen_text.append({"id": batch['ids'][i], "gen_text": gen_text[i]})
            all_gen_text.append({"id": running_id, "gen_text": gen_text[i]})
            running_id += 1

    return all_gen_text 

def clean_up(gen_text, format_cf, tokenizer):
    l1_delim = format_cf['L1_delim']['value'] 

    if l1_delim.strip() != "":
        gen_text = [t.split(l1_delim)[0].strip() for t in gen_text]

    gen_text = [t.replace("</s>", "") for t in gen_text]
    gen_text = [t.replace("<pad>", "") for t in gen_text]
    gen_text = [t.replace(tokenizer.eos_token, "") for t in gen_text]
    return gen_text


def get_lang_from_langcodes(lang, lang_dict):
    if len(lang) > 2:
        key = "FLORES101-code"
    else:
        key = "MM100-code"
    lang = lang_dict[lang_dict[key]==lang]['language'].values[0]
    return lang


def set_lang_delim_tokens(decode_configs, direction): #, prefix):
    # either set as "English" "French" or "[0]" for special prefix

    if "py" in direction:
        return decode_configs

    lang_dict = pd.read_csv("assets/flores_map.csv", sep="\t")
    L1, L2 = direction.split('-')
    L1 = get_lang_from_langcodes(L1, lang_dict)
    L2 = get_lang_from_langcodes(L2, lang_dict)

    decode_configs['header'] = decode_configs['header'].replace("<L1>", L1)
    decode_configs['header'] = decode_configs['header'].replace("<L2>", L2)

    decode_configs['L1_delim']['value'] = decode_configs.L1_delim.value.replace("<L1>", L1)
    decode_configs['L2_delim']['value'] = decode_configs.L2_delim.value.replace("<L2>", L2)
    print("L2_delim:", decode_configs['L2_delim']) 
    return decode_configs


def build_causal_mask_per_batch(model_cf, model, batch):
    """ Construct causal mask on each batch. 
    # There are three options to mask causal attention
        - causal_mask.instructions, 
        - causal_mask.prompts
        - causal_mask.query
    """

    if not model_cf.hack:
        return 

    if "causal_mask" in model_cf:

        batch_mask_from = []
        batch_mask_till = []
        
        for item in range(len(batch['input_ids'])):
            mask_from, mask_till = None, None
            start = batch['input_ids'].shape[1] - batch['input_len'][item]

            if "Llama" in str(model.__class__):
                # Llama needs <s> as the first token
                start = start + 1
            if model_cf.causal_mask.instructions:
                mask_from = start
                mask_till = start + batch['instructions_len'][item]

            if model_cf.causal_mask.prompts:
                if mask_from is None:
                    mask_from = start + batch['instructions_len'][item]
                # this is not a logic bug, it already includes the len of the instructions
                # however the naming needs to be fixed to avoid confusion.
                # code/datasets/prompt_dataset.py: get_prefix
                mask_till = start + batch['prompt_len'][item] + 1

            if model_cf.causal_mask.query:
                if mask_from is None:
                    mask_from = start + batch['prompt_len'][item]
                mask_till = start + batch['prompt_len'][item] + batch['query_len'][item]
           
            batch_mask_from.append(mask_from)
            batch_mask_till.append(mask_till)


        # handle total masking except for single last token
        if "mask_all" in model_cf.causal_mask:
            mask_all = model_cf.causal_mask.mask_all
        else:
            mask_all = False


        construct_mask(model, 
                       model_cf.mask_layer,
                       batch_mask_from, 
                       batch_mask_till,
                       mask_all=mask_all)
    return 

def reset_mask(model):
    if "GPT" in str(model.__class__):
        num_layers = model.config.num_layers
    else:
        num_layers = model.config.num_hidden_layers

    for layer in range(num_layers):
        if "GPTNeo" in str(model.__class__):
            self_attn = model.transformer.h[layer].attn.attention
        elif hasattr(model, "model"):
            self_attn = model.model.layers[layer].self_attn
        elif hasattr(model, "transformer"):
            self_attn = model.transformer.h[layer].self_attention
        else:
            raise Exception("not implemented for model")

        self_attn.mask_prev_positions = False
        self_attn.mask_all = False
        self_attn.mask_from = []
        self_attn.mask_till = []

    return model

def construct_mask(model, mask_layer, batch_mask_from, batch_mask_till, mask_all=False):

    if hasattr(model.config, "num_layers"): 
        num_layers = model.config.num_layers
    else:
        num_layers = model.config.num_hidden_layers

    for layer in range(mask_layer, num_layers):
        if "GPTNeo" in str(model.__class__):
            self_attn = model.transformer.h[layer].attn.attention

        elif hasattr(model, "model"):
            self_attn = model.model.layers[layer].self_attn

        elif hasattr(model, "transformer"):
            self_attn = model.transformer.h[layer].self_attention

        else:
            raise Exception("not implemented for model")
        if mask_all:
            self_attn.mask_all = True
        else:
            self_attn.mask_prev_positions = True
            self_attn.mask_from = batch_mask_from
            self_attn.mask_till = batch_mask_till

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, format_cf, model_cf):
        l1_delim = format_cf['L1_delim']['value'].strip()
        print("l1_delim:", l1_delim)
        # second token for Llama
        if "Llama" in str(model_cf.model_size):
            self.stop_token_ids = tokenizer.encode(l1_delim)[1:]
            self.stop_token_ids = tokenizer.encode("\n")[1:] + self.stop_token_ids
        else:
            self.stop_token_ids = tokenizer.encode(l1_delim)
            self.stop_token_ids = tokenizer.encode("\n") + self.stop_token_ids

        print("stop token ids:", self.stop_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -3].item() == self.stop_token_ids[0] and input_ids[0, -2].item() == self.stop_token_ids[1] and input_ids[0, -1].item() == self.stop_token_ids[2]:
            return True
        return False



