import argparse
import random
import torch
import torch.optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import tokenizer

import util.cfg
import util.seed
from model import MODEL_OPT
from dataset import DSET_OPT
from tokenizer import TKNZR_OPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=int)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    # parser.add_argument('--src', required=True, action='append')
    args = parser.parse_args()

    util.seed.set_seed(seed=args.seed)

    cfg = util.cfg.load(exp_name=args.exp_name)
    tknzr = TKNZR_OPT[cfg.tknzr_type].load(exp_name=cfg.tknzr_exp_name)
    # Load dataset.
    dset = DSET_OPT[cfg.dset](cfg.data, cfg.n_sample)
    batch_title = []
    batch_txt = []
    for i in [23, 19923, 2834, 30000, 81111]:
        data = dset[i]
        batch_title.append(data[0])
        batch_txt.append(data[1])

    model = MODEL_OPT[cfg.model].load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **cfg.__dict__,
    )
    model.eval()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model = model.to(device)

    ignore_tkids = [tknzr.cls_tkid, tknzr.sep_tkid, tknzr.pad_tkid]

    batch_seq = [
        title +
        tknzr.eos_tk +
        content for title,
        content in zip(
            batch_title,
            batch_txt)]

    # shape: (B, S)
    batch_seq = tknzr.batch_enc(batch_seq, max_seq_len=cfg.max_seq_len)

    # shape: (B, S)
    batch_msk_seq = [
        [tknzr.msk_tkid
            if (token not in ignore_tkids) and (random.random() <= 0.15)
            else token
            for token in data
         ]
        for data in batch_seq
    ]

    batch_seq = torch.LongTensor(batch_seq).to(device)
    batch_msk_seq = torch.LongTensor(batch_msk_seq).to(device)
    mask_tks = tknzr.batch_dec(batch_msk_seq.tolist(), rm_sp_tks=False)

    out_probs = model.pred(batch_msk_seq)
    print(f"out_probs: {out_probs.shape}")
    (
        batch_topk_tkid_probs,
        batch_topk_tkid,
    ) = out_probs.topk(
        k=args.k,
        dim=-1,
    )
    print(
        f"batch_topk_tkid_probs, batch_topk_tkid,: {batch_topk_tkid_probs.shape, batch_topk_tkid.shape}")
    batch_pred_tkid_cand_idx = torch.stack(
        [torch.multinomial(B, num_samples=1) for B in batch_topk_tkid_probs]
    )
    print(f"batch_pred_tkid_cand_idx: {batch_pred_tkid_cand_idx.shape}")
    batch_pred_tkid = torch.gather(
        batch_topk_tkid,
        -1,
        batch_pred_tkid_cand_idx,
    )
    print(f"batch_pred_tkid: {batch_pred_tkid.shape}")
    print(batch_msk_seq.type(), batch_msk_seq.device)
    mask = batch_msk_seq == tknzr.msk_tkid
    mask = mask.to(device)
    print(mask.type())
    print(mask.device)
    print(batch_pred_tkid.device)
    print(mask.shape, batch_pred_tkid.shape, batch_msk_seq.shape)
    out_ids = torch.where(
        mask,
        batch_pred_tkid.squeeze(),
        batch_msk_seq
    )

    out_tks = tknzr.batch_dec(out_ids.tolist(), rm_sp_tks=False)
    mask_tks = tknzr.batch_dec(batch_msk_seq.tolist(), rm_sp_tks=False)
    print("Inference:")
    for input, pred in zip(mask_tks, out_tks):
        print(f"input:\n{input}")
        print(f"pred:\n{pred}")


if __name__ == '__main__':
    main()

"""
python infer_mlm_model.py \
    --ckpt 90000 \
    --exp_name char/Transformer_tknzr_news_v2.2_layer3 \
    --k 1 \
    --seed 42 \
"""
