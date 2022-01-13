import argparse
from tokenizer import TKNZR_OPT
from dataset import DSET_OPT
import util.cfg
import util.seed
import util.dist
import numpy as np
import gc
import sqlite3
import os
import random
import pickle
from dataset.mlm import MLMDataset
from tqdm import tqdm
import multiprocessing
from tqdm.contrib.concurrent import process_map

def d2mlm(d, sp_tkids, tknzr, args):
    # print(args)
    title, article = d
    target_tkids = tknzr.enc(
        txt=title,
        txt_pair=article,
        max_seq_len=args.max_seq_len
    )

    mask_tkids = []
    is_mask = []
    mask_span_count = 0

    for tkid in target_tkids:
        mask_span_count -= 1

        # Masking no more than args.p_mask x 100% tokens.
        if sum(is_mask) / len(target_tkids) >= args.p_mask:
            mask_tkids.append(tkid)
            is_mask.append(0)
            continue

        # Skip masking if current token is special token.
        if tkid in sp_tkids:
            mask_tkids.append(tkid)
            is_mask.append(0)
            continue


        # Mask current token based on masking distribution.
        if util.dist.mask(p=args.p_mask):
            # Record how many tokens to be mask (span masking).
            mask_span_count = util.dist.length(
                p=args.p_len,
                max_span_len=args.max_span_len
            )
            mask_tkids.append(tknzr.mask_tkid)
            is_mask.append(1)
            continue

        # Skip masking current token.
        mask_tkids.append(tkid)
        is_mask.append(0)

    # Skipping no masking sample.
    # This is a extreme case which happened with very low possibility.
    if sum(is_mask) == 0:
        return None

    b_mask_tkids = pickle.dumps(mask_tkids)
    b_target_tkids = pickle.dumps(target_tkids)
    b_is_mask = pickle.dumps(is_mask)
    
    return b_mask_tkids,b_target_tkids,b_is_mask

def main():
# if True:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        help='Dataset experiment name.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--n_epoch',
        help='Number of different mask to generate for each sample.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--max_seq_len',
        help='Max sequence length.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--file_name',
        help='SQL file name to read.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        help='Which dataset to use.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--tknzr_exp_name',
        help='Tokenizer experiment name.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--n_sample',
        required=True,
        type=int
    )
    parser.add_argument(
        '--seed',
        help='Different random seeds to generate dataset.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--p_mask',
        help='Masking probability',
        required=True,
        type=float
    )
    parser.add_argument(
        '--p_len',
        help='Mask span length probability',
        required=True,
        type=float
    )
    parser.add_argument(
        '--max_span_len',
        help='Maximum length of masked span.',
        required=True,
        type=int
    )
    args = parser.parse_args()

    # Save configuration.
    util.cfg.save(args)

    cfg = util.cfg.load(exp_name=args.tknzr_exp_name)
    tknzr = TKNZR_OPT[cfg.tknzr](exp_name=args.tknzr_exp_name)
    sp_tkids = [tknzr.cls_tkid, tknzr.sep_tkid, tknzr.pad_tkid]

    # Load dataset.
    dset = DSET_OPT[args.dataset](args.file_name, args.n_sample)

    file_dir = os.path.join(util.path.EXP_PATH, args.exp_name)
    file_path = os.path.join(file_dir, MLMDataset.file_name)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    elif not os.path.isdir(file_dir):
        raise FileExistsError(f'{file_dir} is not a directory.')

    elif os.path.isdir(file_path):
        raise FileExistsError(f'{file_path} is a directory.')

    conn = sqlite3.connect(file_path)
    cur = conn.cursor()
    cur.execute('''
       CREATE TABLE mlm (mask_tkids BLOB, target_tkids BLOB, is_mask BLOB)
    ''')
    print("pairing data....")
    data = list(map(lambda d: (d, sp_tkids, tknzr, args), tqdm(dset)))
    
    # Reinitialize random seeds in each iteration.
    print("creating data....")
    count = 0
    epoch = 0
    util.seed.set_seed(args.seed)
    for seed in [random.randint(0, 2147483647) for _ in range(args.n_epoch)]:
        util.seed.set_seed(seed)
        print(f'epoch: {epoch}')
        print("creating to a list..")
        epoch += 1
        # r= process_map(d2mlm, data, chunksize=100, max_workers=32, smoothing=0)
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            result = pool.starmap(d2mlm, tqdm(data, smoothing=0), chunksize=1024)

        print("write the list to DB..")
        for x in tqdm(result):
            if x == None:
                continue
            b_mask_tkids,b_target_tkids,b_is_mask = x
            count += 1
            cur.execute('INSERT INTO mlm VALUES (?, ?, ?)',
                (b_mask_tkids, b_target_tkids, b_is_mask))
            if count == 100000:
                conn.commit()
                count = 0
        del result
        gc.collect()
        # print(r[-1])
        #desc=f'epoch: {epoch}',
        # print(len(r))


    conn.commit()
    conn.close()


if __name__ == '__main__':
   main()


"""
numactl --membind 1 --cpunodebind 1 \
python create_mlm_dataset.py \
    --exp_name "mask_data_merged_2M_1" \
    --dataset news \
    --file_name "merged_cna_ettoday_storm.db" \
    --max_seq_len 400 \
    --n_epoch 1 \
    --n_sample -1 \
    --seed 12 \
    --p_mask 0.15 \
    --p_len 0.2 \
    --max_span_len 5 \
    --tknzr_exp_name "tknzr_sentPiece_3w5"
"""
