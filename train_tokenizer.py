import argparse

import util.cfg
import util.seed
from dataset import DSET_OPT
from tokenizer import TKNZR_OPT
import os


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dset', required=True, type=str)
    parser.add_argument('--tknzr', required=True, type=str)
    parser.add_argument('--is_uncased', action='store_true')
    parser.add_argument('--max_vocab', required=True, type=int)
    parser.add_argument('--min_count', required=True, type=int)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    # Save configuration.
    util.cfg.save(args)

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    # Load dataset.
    dset = DSET_OPT[args.dset](args.data, args.n_sample)

    # Load tokenizer.
    tknzr = TKNZR_OPT[args.tknzr](**args.__dict__)

    # Train tokenizer.
    batch_txt = []
    for title, article in dset:
        batch_txt.append(title)
        batch_txt.append(article)

    # Build vocab on top of source and target text.
    tknzr.build_vocab(batch_txt)

    # Save trained tokenizer.
    tknzr.save(exp_name=args.exp_name)


if __name__ == '__main__':
    main()


"""
python train_tokenizer.py \
  --exp_name tknzr_news_min6_v2.3 \
  --data news_v2.3.db \
  --dset news \
  --tknzr char \
  --max_vocab -1 \
  --min_count 6 \
  --n_sample -1 \
  --is_uncased \
  --seed 42
"""
