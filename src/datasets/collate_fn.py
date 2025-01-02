#!/usr/bin/python3
# Author: Suzanna Sia

import torch


class CollateFn:
    def __init__(self, tokenizer, cuda=True):
        self.tokenizer = tokenizer

    def get_encoding_length(self, sequence):
        encoded = self.tokenizer.batch_encode_plus(sequence, padding=True, return_tensors='pt')
        enc_ids = encoded['input_ids']
        enc_mask = encoded['attention_mask']
        if torch.cuda.is_available():
            enc_ids = enc_ids.cuda()
            enc_mask = enc_mask.cuda()
        return enc_ids, enc_mask

    def __call__(self, batch):

        inputs = [b['input'] for b in batch]
        instructions = [b['instructions'] for b in batch]
        prompts = [b['prompt'] for b in batch]
        targets = [b['target'] for b in batch]
        queries = [b['query'] for b in batch]
        queries_raw = [b['query_raw'] for b in batch]

        ids = [b['id'] for b in batch]

        input_ids, input_mask = self.get_encoding_length(inputs)
        prompt_ids, prompt_mask = self.get_encoding_length(prompts)
        instructions_ids, instructions_mask = self.get_encoding_length(instructions)
        target_ids, target_mask = self.get_encoding_length(targets)
        query_ids, query_mask = self.get_encoding_length(queries)
        query_raw_ids, _ = self.get_encoding_length(queries_raw)

        input_len = [len(self.tokenizer.encode(b['input'])) for b in batch]
        instructions_len = [len(self.tokenizer.encode(b['instructions'])) for b in batch]
        prompt_len = [len(self.tokenizer.encode(b['prompt'])) for b in batch]
        target_len = [len(self.tokenizer.encode(b['target'])) for b in batch]
        query_len = [len(self.tokenizer.encode(b['query'])) for b in batch]


        return {"ids":ids, 
                "inputs":inputs,
                "input_ids": input_ids, 
                "input_mask": input_mask,
                "input_len": input_len,

                "instructions": instructions,
                "instructions_ids": instructions_ids,
                "instructions_len": instructions_len,
                "instructions_mask": instructions_mask,
 
                "prompts": prompts,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "prompt_len": prompt_len,

                "targets": targets,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "target_len": target_len,

                "queries": queries,
                "query_ids": query_ids,
                "query_mask": query_mask,
                "query_len": query_len,
                "queries_raw": queries_raw,
                "query_raw_ids": query_raw_ids}

