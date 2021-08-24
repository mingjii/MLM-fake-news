import argparse

import util.cfg
import util.path
import util.seed
import re
import os
from tqdm import tqdm
from dataset import DSET_OPT
from tokenizer.sentence_piece import Tknzr_sentPiece


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dset', required=True, type=str)
    parser.add_argument('--tknzr', required=True, type=str)
    parser.add_argument('--is_uncased', action='store_true')
    parser.add_argument('--vocab_size', required=True, type=int)
    parser.add_argument('--char_coverage', required=True, type=float)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    data_version = re.match(r'news_(v\d+.\d+).db', args.data)
    if data_version:
        data_version = data_version.group(1)
    else:
        raise ValueError('`exp_name` must has prefix `news_`.')

    n_sample = f'_{args.n_sample}' if args.n_sample > 0 else ''
    rawData_name = f'rawtext_{data_version}{n_sample}.txt'

    rawData_path = os.path.join(util.path.DATA_PATH, rawData_name)

    # Save configuration.
    util.cfg.save(args)

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    # Load tokenizer.
    tknzr = Tknzr_sentPiece(**args.__dict__)

    if not os.path.exists(rawData_path):
        # Load dataset.
        print("loading data......")
        dset = DSET_OPT[args.dset](args.data, args.n_sample)

        # create raw data.
        print("creating raw data......")
        with open(rawData_path, 'w', encoding='utf-8') as f:
            for title, article in tqdm(dset):
                f.write(title + '\n')
                f.write(article + '\n')
    print("Raw data create complete.")

    # Build vocab on top of source and target text.
    tknzr.train(
        data_file=rawData_name,
        exp_name=args.exp_name,
    )


if __name__ == '__main__':
    main()


"""
python train_tokenizer_sentencePiece.py \
  --exp_name tknzr_sentPiece_v2.3_test \
  --data news_v2.3.db \
  --tknzr sentencePiece \
  --dset news \
  --vocab_size 4000 \
  --char_coverage 0.9995 \
  --n_sample 1000 \
  --is_uncased \
  --seed 42
"""
