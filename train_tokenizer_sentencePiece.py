import argparse

import util.cfg
import util.path
import util.seed
import re
import os
from tqdm import tqdm
from dataset import DSET_OPT
# from tokenizer.sentence_piece_hg import Tknzr_sentPiece
from tokenizer import TKNZR_OPT


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dset', required=True, type=str)
    parser.add_argument('--tknzr', required=True, type=str)
    parser.add_argument('--is_uncased', action='store_true')
    parser.add_argument('--vocab_size', required=True, type=int)
    parser.add_argument('--char_coverage', required=False, type=float)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    # data_version = re.match(r'news_(v\d+.\d+).db', args.data)
    # if data_version:
    #     data_version = data_version.group(1)
    # else:
    #     raise ValueError('`exp_name` must has prefix `news_`.')

    # n_sample = f'_{args.n_sample}' if args.n_sample > 0 else ''
    # rawData_name = f'rawtext_{data_version}{n_sample}.txt'
    if args.n_sample > 0 and args.n_sample < 10000:
        rawData_name = f'test{args.n_sample}_{args.data[:-3]}_sentencetext.txt'
    else:
        rawData_name = f'{args.data[:-3]}_sentencetext.txt'
    rawData_path = os.path.join(util.path.DATA_PATH, rawData_name)

    # Save configuration.
    util.cfg.save(args)

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    if not os.path.exists(rawData_path):
        # Load dataset.
        print("loading data......")
        dset = DSET_OPT[args.dset](args.data, args.n_sample)
        sentence_pattern = re.compile(r'[，|,|。|：|:|；|;|！|!|？|?]')
        # create raw data.
        print("creating raw data......")
        with open(rawData_path, 'w', encoding='utf-8') as f:
            for title, article in tqdm(dset):
                f.write(title + '\n')
                for sentence in sentence_pattern.split(article):
                    if sentence != "":
                        f.write(sentence + '\n')
    print("Raw data create complete.")
    # exit()
    # Build vocab on top of source and target text.
    # tknzr.train(
    #     data_file=rawData_name,
    #     exp_name=args.exp_name,
    # )
    
    TKNZR_OPT[args.tknzr].train(
        exp_name=args.exp_name,
        files=[rawData_path],
        vocab_size=args.vocab_size,
    )


if __name__ == '__main__':
    main()


"""
python train_tokenizer_sentencePiece.py \
  --exp_name test_tknzr_sentPiece \
  --data merged_cna_ettoday_storm.db \
  --tknzr sentencePiece \
  --dset news \
  --vocab_size 4500 \
  --n_sample 2000 \
  --is_uncased \
  --seed 42
"""
