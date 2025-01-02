#!/usr/bin/python3
# Author: Suzanna Sia

from torch.utils.data import Dataset

class PromptsDataset(Dataset):
    def __init__(self, 
                format_cf, 
                ds_promptbank, 
                ds_test, 
                seed=0,
                ntest=-1,
                nprompts=5,
                skey="source",
                tkey="target",
                sample_on_new=False):
        # take in two datasets, 
        # the first will be used to prompt, the second will be used to test
        super().__init__()
        self.nprompts = nprompts
        self.seed = seed
        self.calc_token_count(ds_promptbank)
        self.calc_token_count(ds_test)
        self.ds_promptbank = ds_promptbank
        self.ds_test = ds_test
        self.printed = False # just to print once

        self.q = format_cf['L1_delim']['value']
        self.a = format_cf['L2_delim']['value']
        self.eos = format_cf['eos']
        self.sep = format_cf['sep']
        self.header = format_cf['header']
        self.skey = skey
        self.tkey = tkey

        self.sample_on_new = sample_on_new # sample new prefix for every item
        self.prefix = self.get_prefix()
        self.ntest = ntest

    def calc_token_count(self, ds):
        ds.df['src_wc'] = ds.df['source'].apply(lambda x: len(x.split()))
        ds.df['target_wc'] = ds.df['target'].apply(lambda x: len(x.split()))
        ds.df['src_target_wc'] = ds.df['src_wc'] + ds.df['target_wc'] 

    def get_vals(self):
        ds = self.ds_promptbank
        if self.sample_on_new:
            vals = ds.df.sample(n=self.nprompts) #, random_state=self.seed)
        else:
            vals = ds.df.sample(n=self.nprompts, random_state=self.seed)
        vals = vals[[self.skey, self.tkey]]
        return vals

    def get_prefix(self):
        # we only need to get the prefix once.
        vals = self.get_vals()
        vals = vals.values
        vals = [(v[0].strip(), v[1].strip()) for v in vals]
        prefix = f"{self.sep}".join([f"{self.q}{v[0]}{self.a}{v[1]} " for v in vals])
        # to make sure new line written correctly. 
        if prefix == "":
            prefix = f"{self.header}{prefix}"
        else:
            prefix = f"{self.header}{self.sep}{prefix}"
        return prefix

    def __getitem__(self, i):

        # if sample prefix, we keep sampling a new one everytime theres a new item.
        if self.sample_on_new:
            self.prefix = self.get_prefix()

        source = self.ds_test[i][self.skey]
        #if source.strip()[-1] not in ['!', '.', '?']:
        #    source = source + "."
        
        query = f"{self.q}{source}{self.a}" 
        total_input = f"{self.prefix}{self.sep}{query}"

        if not self.printed:
            print(total_input)
            self.printed = True

        item = {"id": self.ds_test[i]['id'],
                "instructions": self.header,
                "input": total_input,
                "prompt": self.prefix,
                "query": query,
                "query_raw": source,
                "target": self.ds_test[i]['target'] + self.eos}

        return item 

    def __len__(self):
        if self.ntest==-1:
            return len(self.ds_test)
        return min(len(self.ds_test), self.ntest)

