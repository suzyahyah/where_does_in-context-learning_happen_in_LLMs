#!/usr/bin/python3
# Author: Suzanna Sia

import pandas as pd
import os
import json
from torch.utils.data import Dataset


# helper function to map to the datasets in this file.
def get_fn_dataset(dataname, mode, direction="en-fr", data_path="data"):
    name2dataset = {
        "FLORES": FLORESdataset,
        "MBPP": MBPPdataset,
        "HEVAL": HEVALdataset
    }

    if dataname not in name2dataset:
        raise Exception(dataname, "not recognised. Only has {name2dataset.keys()}")

    DATASET = name2dataset[dataname]

    if dataname == "FLORES":
        # flores has no train set, so we sample prompts from the dev set. 
        splits_ = {
            "train": "dev",
            "valid": "dev",
            "test": "devtest"
        }
        return DATASET(splits_[mode], direction)

    args = {"dataname":dataname, "mode":mode, "direction":direction}
    fn = data_path.format(**args)
    dataset = DATASET(fn, direction)
    print(f'Loaded {mode} {dataname} file: {fn}')
    return dataset



class BitextDataset(Dataset):
    def __getitem__(self, i):
        item = self.df.iloc[i]
        return item

    def __len__(self):
        return len(self.df)


class FLORESdataset(BitextDataset):
    def __init__(self, mode="", direction=""):
        super().__init__()
        self.name = "FLORES"
        self.direction = direction
        L1, L2 = direction.split('-')

        # lang conversion
        flores_map = pd.read_csv('assets/flores_map.csv', header=0, sep="\t")
        if len(L1) < 3: 
            L1 = flores_map[flores_map['MM100-code'] == L1]['FLORES101-code'].values[0]
            L2 = flores_map[flores_map['MM100-code'] == L2]['FLORES101-code'].values[0]

        fn1 = f"data/FLORES/flores101_dataset/{mode}/{L1}.{mode}"
        fn2 = f"data/FLORES/flores101_dataset/{mode}/{L2}.{mode}"
        L1_data, L2_data = self.construct_from_bitext(fn1, fn2)
        
        df = pd.DataFrame()
        df['source'] = L1_data
        df['target'] = L2_data

        df['id'] = list(range(len(L1_data)))
        self.df = df
        print(f"FLORES {mode=} len lines:{len(self.df)}") 

    @staticmethod
    def get_lang_from_langcodes(lang):
        lang_dict = pd.read_csv("assets/flores_map.csv", sep="\t")
        if len(lang) > 2:
            key = "FLORES101-code"
        else:
            key = "MM100-code"

        lang = lang_dict[lang_dict[key]==lang]['language'].values[0]
        return lang

    def construct_from_bitext(self, fn1, fn2):
        with open(fn1, 'r') as f:
            L1_data = f.readlines()

        with open(fn2, 'r') as f:
            L2_data = f.readlines()
        
        L1_data = [l1.strip() for l1 in L1_data]
        L2_data = [l2.strip() for l2 in L2_data]
        return L1_data, L2_data




class MBPPdataset(BitextDataset):
    def __init__(self, fn, direction):
        super().__init__()
        self.name = "MBPP"
        datas = []
        with open(fn, 'r') as f:
            for line in f:
                datas.append(json.loads(line))
        datas = pd.DataFrame(datas)
        datas = datas.rename(columns={"task_id": "id", "text": "source", "code": "target"})
        self.df = datas


class HEVALdataset(BitextDataset):
    def __init__(self, fn, direction):
        super().__init__()
        self.name = "HEVAL"
        datas = []
        with open(fn, 'r') as f:
            for line in f:
                datas.append(json.loads(line))
        datas = pd.DataFrame(datas)
        datas = datas.rename(columns={"task_id": "id", "prompt": "source",
            'canonical_solution':"target"})
        datas['source'] = datas['source'].apply(lambda x: x.split('"""')[1].strip() if '"""' in x else x)
        self.df = datas
