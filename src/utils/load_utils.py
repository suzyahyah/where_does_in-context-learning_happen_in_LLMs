#!/usr/bin/python3

import os
from src.utils.nvidia_utils import gpu_util
from src.model_hacks import (
    gptneo_model_hack,
    bloom_model_hack,
    starcoder2_model_hack,
    llama_model_hack
)



from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
#from peft import PeftModel
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def get_model_class(model_size, args_model):

    if args_model.hack:
        cls_name = None
        name2class = {
                "gptn": gptneo_model_hack.GPTNeoForCausalLMHack,
                "bloom": bloom_model_hack.BloomForCausalLMHack,
                "starcoder2": starcoder2_model_hack.Starcoder2ForCausalLMHack,
                "llama": llama_model_hack.LlamaForCausalLMHack
        }
        for key in name2class:
            if key in model_size:
                cls_name = name2class[key]

        if not cls_name:
            raise NotImplementedError(f"Model Hack not implemented for: {model_size}")
    else:
        cls_name = AutoModelForCausalLM

    print("Loading model class:", str(cls_name.__name__))
    return cls_name


def load_if_hack(cls_name, og_fol, args_model):
    quant_config = None
    if "7b" in args_model.model_size or "7B" in args_model.model_size:
        quant_config = BitsAndBytesConfig(load_in_8bit=True,
                                          llm_int8_enable_fp32_cpu_offload=True)

    if args_model.hack:
        model = cls_name.from_pretrained(og_fol, device_map='auto',
               ignore_mismatched_sizes=True, quantization_config=quant_config)
    else:
        model = cls_name.from_pretrained(og_fol, quantization_config=quant_config)
    return model

def get_model_path(model_size="gptn2.7B"):
    if "gptn" in model_size:
        size = model_size.replace("gptn", "")
        og_fol = f"EleutherAI/gpt-neo-{size}"
    elif "bloom" in model_size:
        size = model_size.replace("bloom", "")
        og_fol = f"bigscience/bloom-{size}"
    elif "llama" in model_size:
        if "llama3" in model_size:
            og_fol = f"/exp/ssia/projects/meta-llama/Meta-Llama-3-8B-Instruct" 
        else:
            og_fol = f"/exp/ssia/projects/llama/models/{model_size}"

    elif "starcoder2" in model_size:
        og_fol = f'bigcode/{model_size}' 
    else:
        raise Exception("model not specified:", model_size)

    return og_fol

@gpu_util
def get_models(model_size="gptn2.7B", save_fol="", args_model=None):
    #hack=False
    if save_fol != "":
        print(f"loading models from..{save_fol}")

    cls_name = get_model_class(model_size, args_model)
    model_fol = get_model_path(model_size)
    tokenizer = AutoTokenizer.from_pretrained(model_fol)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = load_if_hack(cls_name, model_fol, args_model)
    print("loaded models..")
    if "lora" in args_model.save_fol:
        # instead of model_id should be save_fol?
        model = PeftModel.from_pretrained(model, model_id=args_model.save_fol)
        model = model.merge_and_unload()
        print("Merging lora layer from:", args_model.save_fol)
    return model, tokenizer
