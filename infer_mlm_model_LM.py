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
    dset = MLMDataset(exp_name=exp_cfg.dataset_exp_name,
                      n_sample=exp_cfg.n_sample)
    dset_cfg = util.cfg.load(exp_name=exp_cfg.dataset_exp_name)

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=dset_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr].load(exp_name=tknzr_cfg.exp_name)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    batch_mask_tkids = []
    batch_target_tkids = []
    batch_is_mask = []
    for i in [1, 1001, 2001, 3001, 4001]:
        (mask_tkids, target_tkids, is_mask) = dset[i]
        batch_mask_tkids.append(mask_tkids)
        batch_target_tkids.append(target_tkids)
        batch_is_mask.append(is_mask)

    model, _ = MODEL_OPT[exp_cfg.model].load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **exp_cfg.__dict__,
    )
    model.eval()

    model = model.to(device)

    # B x S
    batch_mask_tkids = torch.LongTensor(batch_mask_tkids).to(device)
    batch_target_tkids = torch.LongTensor(batch_target_tkids).to(device)
    batch_is_mask = torch.BoolTensor(batch_is_mask).to(device)

    output = []
    for mask_tkids, is_mask in zip(batch_mask_tkids, batch_is_mask):
        # S, 1
        output_tkids = mask_tkids.clone()
        for i in range(exp_cfg.max_seq_len):
            if not is_mask[i]:
                continue

            # In: 1, S
            # Out: S, V
            out_probs = model.pred(output_tkids.unsqueeze(0)).squeeze()

            # In: V
            # Out: K
            (
                topk_tkid_probs,
                topk_tkid,
            ) = out_probs[i].topk(
                k=args.k,
                dim=-1,
            )

            # 1
            pred_tkid_cand_idx = torch.multinomial(
                topk_tkid_probs, num_samples=1)

            # 1
            pred_tkid = torch.gather(
                topk_tkid,
                -1,
                pred_tkid_cand_idx,
            )

            output_tkids[i] = pred_tkid

        output.append(output_tkids.tolist())

    batch_out_tks = tknzr.batch_dec(output, rm_sp_tks=False)
    batch_mask_tks = tknzr.batch_dec(
        batch_mask_tkids.tolist(), rm_sp_tks=False)
    batch_target_tks = tknzr.batch_dec(
        batch_target_tkids.tolist(), rm_sp_tks=False)
    mask_count = 0
    mask_acc = 0

    print('<table>')
    for text, target, pred in zip(batch_mask_tks, batch_target_tks, batch_out_tks):
        l_text = tknzr.tknz(text)
        l_target = tknzr.tknz(target)
        l_pred = tknzr.tknz(pred)
        print('<tr><th>text</th><th>target</th><th>pred</th></tr>')
        for a, b, c in zip(l_text, l_target, l_pred,):
            if a == "<mask>":
                mask_count += 1
                if b == c:
                    mask_acc += 1
                else:
                    c += "<ERROR>"

            print('<tr>')
            print(
                f'<td>{html.escape(a)}</td><td>{html.escape(b)}</td><td>{html.escape(c)}</td>')
            print('</tr>')

    print('</table>')

    # print(f"<div>predict mask accuracy:{mask_acc/mask_count}, number of error:{mask_count-mask_acc}</div>")
    # print(f"<div>predict all tokens accuracy:{all_acc/count}, number of error:{count-all_acc}</div>")
    # r_text = text.replace("<mask>", "<b style=\"color:lightgreen\">m</b>")
    # print(f'<div style="border: 1px solid black;">{r_text}</div>')
    # print(f'<div style="border: 1px solid black;">{target}</div>')
    # print(f'<div style="border: 1px solid black;">{pred}</div>')
    # print(f"input:\n{text}")
    # print(f"target:\n{target}")
    # print(f"pred:\n{pred}")
    # print('</div>')


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python infer_mlm_model_LM.py \
    --ckpt 200000 \
    --exp_name n10_m7_p10_v2.3 \
    --k 1 \
    --seed 42 >>alldata_LM.html
"""
