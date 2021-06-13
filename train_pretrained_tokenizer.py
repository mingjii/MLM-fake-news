import argparse
import re
import os
from tokenizer import pretrained_Tknzr
import util.cfg
from dataset import DSET_OPT


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dset', required=True, type=str)
    parser.add_argument('--n_sample', required=True, type=int)
    args = parser.parse_args()

    # Save configuration.
    util.cfg.save(args)

    # Load dataset.
    dset = DSET_OPT[args.dset](args.data, args.n_sample)
    # tiltle + content
    data = [x + y for x, y in dset]

    # Pattern for all preprocessed special tokens.
    pre_pattern = re.compile(r'<(en|num|unk|(loc|org|per)\d+)>')

    # Pattern for model training special tokens.
    model_pattern = re.compile(r'<(cls|sep|pad|unk|mask)>')
    additional_tks = []
    for text in data:
        match = pre_pattern.findall(text)
        for tk in match:
            if tk not in additional_tks:
                additional_tks.append(tk)

        match = model_pattern.findall(text)
        for tk in match:
            if tk not in additional_tks:
                additional_tks.append(tk)

    tknzr = pretrained_Tknzr(
        additional_tokens=additional_tks,
    )
    tknzr.save(args.exp_name)


if __name__ == "__main__":
    main()

"""
python train_pretrained_tokenizer.py \
    --exp_name pretknzr_news_v2.3 \
    --data news_v2.3.db \
    --dset news \
    --n_sample -1
"""
