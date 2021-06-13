import argparse
import random
import torch
import torch.optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

import util.path
import util.cfg
import util.seed
from model import MODEL_OPT
from dataset import DSET_OPT
from tokenizer import TKNZR_OPT


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--tknzr_exp_name', required=True, type=str)
    parser.add_argument('--tknzr_type', required=True, type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--max_seq_len', required=True, type=int)
    parser.add_argument('--n_hid_lyr', required=True, type=int)
    parser.add_argument('--n_epoch', required=True, type=int)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--d_emb', required=True, type=int)
    parser.add_argument('--d_hid', required=True, type=int)
    parser.add_argument('--p_emb', required=True, type=float)
    parser.add_argument('--p_hid', required=True, type=float)
    parser.add_argument('--ckpt_step', required=True, type=int)
    parser.add_argument('--log_step', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--dset', required=True, type=str)
    parser.add_argument('--beta1', required=True, type=float)
    parser.add_argument('--beta2', required=True, type=float)
    parser.add_argument('--eps', required=True, type=float)
    parser.add_argument('--max_norm', required=True, type=float)
    parser.add_argument('--wd', required=True, type=float)
    args = parser.parse_args()

    # Save configuration.
    util.cfg.save(args)

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    # Load dataset.
    dset = DSET_OPT[args.dset](args.data, args.n_sample)

    # Create data loader.
    dldr = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=len(os.sched_getaffinity(0)),
    )

    # Load tokenizer.
    tknzr = TKNZR_OPT[args.tknzr_type].load(exp_name=args.tknzr_exp_name)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load model.
    model = MODEL_OPT[args.model](tknzr=tknzr, **args.__dict__)
    model = model.train()

    # Move model to running device.
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args.wd,
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Create optimizer.
    optim = torch.optim.AdamW(
        optim_group_params,
        betas=(args.beta1, args.beta2),
        lr=args.lr,
        eps=args.eps,
    )

    # Loggin.
    log_path = os.path.join(util.path.LOG_PATH, args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    # Log performance.
    pre_avg_loss = 0.0
    avg_loss = 0.0

    # Global optimization step.
    step = 0

    ignore_tkids = [tknzr.cls_tkid, tknzr.sep_tkid, tknzr.pad_tkid]
    for epoch in range(args.n_epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {epoch}, loss: {pre_avg_loss:.6f}',
        )
        for batch_title, batch_content in tqdm_dldr:
            batch_seq = [
                title +
                tknzr.sep_tk +
                content for title,
                content in zip(
                    batch_title,
                    batch_content)]

            # shape: (B, S)
            batch_seq = tknzr.batch_enc(
                batch_seq, max_seq_len=args.max_seq_len)

            # shape: (B, S)
            batch_msk_seq = [
                [tknzr.mask_tkid
                    if (token not in ignore_tkids) and (random.random() <= 0.15)
                    else token
                    for token in data
                 ]
                for data in batch_seq
            ]

            batch_seq = torch.LongTensor(batch_seq).to(device)
            batch_msk_seq = torch.LongTensor(batch_msk_seq).to(device)

            loss = model.loss_fn(
                batch_msk_seq=batch_msk_seq,
                batch_seq=batch_seq)
            avg_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.max_norm,
            )

            optim.step()
            optim.zero_grad()

            step += 1

            if step % args.ckpt_step == 0:
                model.save(ckpt=step, exp_name=args.exp_name)

            if step % args.log_step == 0:
                avg_loss = avg_loss / args.log_step

                # Log on tensorboard
                writer.add_scalar(
                    f'loss/{args.exp_name}',
                    avg_loss,
                    step,
                )

                # Refresh log performance.
                pre_avg_loss = avg_loss
                avg_loss = 0.0

                tqdm_dldr.set_description(
                    f'epoch: {epoch}, loss: {pre_avg_loss:.6f}'
                )

    # Save last checkpoint.
    model.save(ckpt=step, exp_name=args.exp_name)

    # Close tensorboard logger.
    writer.close()


if __name__ == '__main__':
    main()
