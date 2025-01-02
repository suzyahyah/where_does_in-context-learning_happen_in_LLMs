#!/usr/bin/python3
# Author: Suzanna Sia
# pylint: disable=C0303,C0103

### Standard imports
import argparse
from omegaconf import OmegaConf

### Third Party imports
from torch.utils.data import DataLoader


### Local/Custom imports
from src.utils import utils, build, io_utils
from src.datasets.collate_fn import CollateFn


def run(args, cfp, do_baseline=False):

    args.format = utils.set_lang_delim_tokens(args.format,args.data.direction)

    promptbank, test_dataset = build.build_datasets_for_prompt(args.data)
    prompt_ds = build.build_prompt_dataset(args, promptbank, test_dataset)
    model, tokenizer = build.build_model_tok(args.model)
    model.training = False

    collate_fn = CollateFn(tokenizer)
    dataloader = DataLoader(prompt_ds,
                            collate_fn=collate_fn,
                            batch_size=args.generator.batch_size)

    if do_baseline:
        print("save to:", cfp['gen_fn_baseline'].format(**args))
        if not args.eval_only:
            all_gen_text = utils.generation(args.hf_generator_configs,
                                            args.format,
                                            args.model,
                                            model,
                                            tokenizer,
                                            dataloader)

        io_utils.save_and_eval(cfp, args, all_gen_text)

    else:
        # do layering experiments here.
        print("save to:", cfp['gen_fn'].format(**args))
        if hasattr(model.config, "num_layers"):
            num_layers = model.config.num_layers
        elif hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers

        for layer in range(num_layers):
            args.model.mask_layer = layer
            if not args.eval_only:
                all_gen_text = utils.generation(args.hf_generator_configs,
                                                args.format,
                                                args.model,
                                                model,
                                                tokenizer,
                                                dataloader)

                io_utils.save_test_prompts_used(args, cfp, prompt_ds) 
            io_utils.save_and_eval(cfp, args, all_gen_text)
            model = utils.reset_mask(model)

    # Run the baseline
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--seed', default=0, type=int)
    argparser.add_argument('--do_analysis', default="")
    argparser.add_argument('--do_baseline', action="store_true")
    argparser.add_argument('--eval_only', action="store_true")

    argparser.add_argument('--training_cf', default='configs/training/default.yaml')
    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml')
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random.yaml')

    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--heval', action="store_true")
    argparser.add_argument('--file_paths_cfg', default="")

    args, uk_args = argparser.parse_known_args()

    args = io_utils.merge_omegaconf_w_argparse(args, uk_args)
    cfp = OmegaConf.load(args.file_paths_cfg)

    run(args, cfp, do_baseline=args.do_baseline)
