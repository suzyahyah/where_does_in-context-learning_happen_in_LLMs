#!/usr/bin/env bash
# Author: Suzanna Sia


MBPP_PATH=https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl 
HEVAL_PATH=https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

wget $MBPP_PATH
wget $HEVAL_PATH

gunzip HumanEval.jsonl.gz

mv mbbp.jsonl data
mv HumanEval.jsonl data

sed -n '1,800p' data/mbpp.jsonl > data/MBPP/train/en-py.tsv
sed -n '801,974p' data/mbpp.jsonl > data/MBPP/valid/en-py.tsv
cp data/HumanEval.jsonl data/HEVAL/test/en-py.tsv
