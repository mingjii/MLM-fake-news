import argparse
import random
import torch
import torch.optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import tokenizer
import html
import util.cfg
import util.seed
from model import MODEL_OPT
from dataset.mlm import MLMDataset
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

    exp_cfg = util.cfg.load(exp_name=args.exp_name)
    # Load dataset and dataset config.
    # dset = MLMDataset(exp_name=exp_cfg.dataset_exp_name,
    #                   n_sample=exp_cfg.n_sample)
    dset = MLMDataset(exp_name=exp_cfg.dataset_exp_name,
                      n_sample=1000)
    dset_cfg = util.cfg.load(exp_name=exp_cfg.dataset_exp_name)

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=dset_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr].load(exp_name=tknzr_cfg.exp_name)

    def collate_fn(batch):
        batch_mask_tkids = []
        batch_target_tkids = []
        batch_is_mask = []
        for mask_tkids, target_tkids, is_mask in batch:
            batch_mask_tkids.append(mask_tkids)
            batch_target_tkids.append(target_tkids)
            batch_is_mask.append(is_mask)

        return batch_mask_tkids, batch_target_tkids, batch_is_mask

    # Create data loader.
    dldr = torch.utils.data.DataLoader(
        dset,
        batch_size=exp_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    model, _ = MODEL_OPT[exp_cfg.model].load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **exp_cfg.__dict__,
    )
    model.eval()
    model = model.to(device)

    mask_count = 0
    mask_acc = 0
    topn_acc = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    for batch_mask_tkids, batch_target_tkids, batch_is_mask in tqdm(dldr):
        batch_mask_tkids = torch.LongTensor(batch_mask_tkids).to(device)
        batch_target_tkids = torch.LongTensor(batch_target_tkids).to(device)
        batch_is_mask = torch.BoolTensor(batch_is_mask).to(device)

        # B, S, V
        out_probs = model.pred(batch_mask_tkids)

        # BxS, V
        target = batch_target_tkids.view(-1)
        is_mask = batch_is_mask.view(-1)
        for n in topn_acc.keys():
            (
                batch_topk_tkid_probs,
                batch_topk_tkid,
            ) = out_probs.topk(
                k=n,
                dim=-1,
            )
            # BxS, n
            topn_tkids = batch_topk_tkid.view(-1, n)
            for t, m, topn in zip(target, is_mask, topn_tkids):
                if m and (t in topn):
                    topn_acc[n] += 1

        # print(f"out_probs's shape: {out_probs.shape}")
        (
            batch_topk_tkid_probs,
            batch_topk_tkid,
        ) = out_probs.topk(
            k=args.k,
            dim=-1,
        )
        # B, S, K
        # print(
        #     f"batch_topk_tkid_probs's shape, batch_topk_tkid's shape,: {batch_topk_tkid_probs.shape, batch_topk_tkid.shape}")
        B = batch_topk_tkid_probs.shape[0]
        S = batch_topk_tkid_probs.shape[1]

        batch_pred_tkid_cand_idx = torch.stack(
            [torch.multinomial(BS, num_samples=1)
             for BS in batch_topk_tkid_probs.view(-1, args.k)]
        )
        # print(f"batch_pred_tkid_cand_idx's shape: {batch_pred_tkid_cand_idx.shape}")
        batch_pred_tkid = torch.gather(
            batch_topk_tkid,
            -1,
            batch_pred_tkid_cand_idx.view(B, S, 1),
        )
        # print(f"batch_pred_tkid's shape: {batch_pred_tkid.shape}")

        # print(batch_is_mask.type())
        # print(batch_is_mask.device)
        # print(batch_pred_tkid.device)
        # print(batch_is_mask.shape, batch_pred_tkid.shape, batch_target_tkids.shape)
        out_ids = torch.where(
            batch_is_mask,
            batch_pred_tkid.squeeze(),
            batch_target_tkids
        )

        # out_tks = tknzr.batch_dec(out_ids.tolist(), rm_sp_tks=False)
        # ori_out_tks = tknzr.batch_dec(batch_pred_tkid.squeeze().tolist(), rm_sp_tks=False)
        # mask_tks = tknzr.batch_dec(batch_mask_tkids.tolist(), rm_sp_tks=False)
        # target_tks = tknzr.batch_dec(batch_target_tkids.tolist(), rm_sp_tks=False)
        mask_count += batch_is_mask.sum().item()
        mask_acc += torch.sum((out_ids == batch_target_tkids)
                              * batch_is_mask).item()

    print(f"<div>{args.exp_name}</div>")
    print(
        f"<div>predict mask accuracy:{mask_acc/mask_count}, number of error:{mask_count-mask_acc}</div>")
    print(f"<div>Top n accuracy</div>")
    for n, count in topn_acc.items():
        print(f"<div>n={n}:{count/mask_count}</div>")


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python eval_mlm_model.py \
    --ckpt 200000 \
    --exp_name n10_m7_p10_v2.3 \
    --k 1 \
    --seed 42 >>alldata_LM.html
"""
