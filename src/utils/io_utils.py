#!/usr/bin/python3
# Author: Suzanna Sia

### Standard imports
import os
import json
import pathlib
import pandas as pd
from omegaconf import OmegaConf
from src.evaluate import bleu as eval_bleu
from src.evaluate.evaluate_heval import evaluate_functional_correctness

mkpath = lambda x: pathlib.Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True)


def save_test_prompts_used(args, cfg, prompt_ds):

    ds2 = prompt_ds.ds_test
    args.data.source, args.data.target = args.data.direction.split('-')

    test_source_fn = cfg['test_source_fn'].format(**args)
    test_target_fn = cfg['test_target_fn'].format(**args)

    mkpath(test_source_fn)
    mkpath(test_target_fn)

    # change this to dataframes

    with open(test_source_fn, 'w') as f:
        f.write("\n".join(ds2.df['source']))
    with open(test_target_fn, 'w') as f:
        f.write("\n".join(ds2.df['target']))

    parallel_csv = cfg['test_parallel_fn'].format(**args)
    mkpath(parallel_csv)
    print("parallel csv:", parallel_csv)
    ds2.df.to_csv(parallel_csv, index=False, sep="\t", header=True)  
    # get vals for everything
    # save the vals in used_prompts_fn
    
    # save used prompts
    used_prompts_fn = cfg['used_prompts_fn'].format(**args)
    mkpath(used_prompts_fn)

    with open(used_prompts_fn, 'w') as f:
        f.write(prompt_ds[0]['prompt'])


def save_and_eval(cfp, args, all_gen_text):


    if args.do_baseline:
        hyp_fn = cfp['gen_fn_baseline'].format(**(args))
    else:
        hyp_fn = cfp['gen_fn'].format(**(args))

    #if args.data.ntest != -1:
    #    hyp_fn = hyp_fn + ".temp"

    if args.data.dev:
        hyp_fn = hyp_fn + ".dev"

    mkpath(hyp_fn) 

    if not args.eval_only:
        pd.DataFrame(all_gen_text).to_csv(hyp_fn, index=False, sep="\t")

    if "py" in args.data.direction:
        score = evaluate_functional_correctness(hyp_fn,
                k=[1],
                problem_file="data/HumanEval.jsonl")
        if args.do_baseline:
            res_fn = cfp['res_fn_baseline'].format(**args)

        else:
            res_fn = cfp['res_fn'].format(**args)

        dir_path = os.path.dirname(res_fn)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(res_fn, 'w') as f:
            json.dump(score, f)

    else:
        ref_fn = cfp['test_parallel_fn'].format(**args)
        eval_bleu.main(hyp_fn, ref_fn)


def merge_omegaconf_w_argparse(args, uk_args, verbose=True):

    # merging omega conf with other things
    # reading in from config file

    config_files = []
    attributes = vars(args)
    for key in attributes.keys():
        if key.endswith('_cf'):
            config_files.append(OmegaConf.load(attributes[key]))
    known_args = OmegaConf.merge(*config_files)


    new_args = {}
    # reading unknown args 

    for i, uk in enumerate(uk_args):
        if uk.startswith('--') and uk != "nproc_per_node":
            uk = uk.replace('--', '')
            namespace = "" 

            if "." not in uk:
                print(f"missing namespace for argument {uk}")

            if uk.count('.') == 1:
                namespace, arg = uk.split('.')

            elif uk.count('.') == 2:
                namespace, arg, arg2 = uk.split('.')

            if namespace not in known_args:
                raise Exception("Unknown namespace:", namespace)

            elif namespace not in new_args:
                new_args[namespace] = {}

            if arg not in known_args[namespace]:
                raise Exception(f"Unknown arg {arg} in namespace {namespace}")

            if not uk_args[i+1].startswith('--'):
                
                if uk_args[i+1].isdigit():
                    val = int(uk_args[i+1])
                else:
                    try:
                        val = float(uk_args[i+1])
                    except:
                        val = uk_args[i+1]

                #print(namespace, arg, val)
                if uk.count('.') == 1:
                    new_args[namespace][arg] = val
                elif uk.count('.') == 2:
                    if arg not in new_args[namespace]:
                        new_args[namespace][arg] = {}
                    new_args[namespace][arg][arg2] = val
            else:
                if uk.count('.') == 1:
                    new_args[namespace][arg] = True
                elif uk.count('.') == 2:
                    if arg not in new_args[namespace]: 
                        new_args[namespace][arg] = {}
                    new_args[namespace][arg][arg2] = True

    result = OmegaConf.merge(known_args, new_args)
    if verbose:
        print(OmegaConf.to_yaml(result))
    if "format" in result:
        result.format.direction = result.data.direction 
        result.format.domain = result.data.testset

    args_ = vars(args)
    for k in args_:
       result[k] = args_[k]

    return result

def get_default_argparser():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', default=0)
    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random.yaml')
    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml')
    argparser.add_argument('--training_cf', default='configs/training/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--file_paths_cfg', default="")

    return argparser
