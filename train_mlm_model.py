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
from dataset.mlm import MLMDataset
from tokenizer import TKNZR_OPT


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--dataset_exp_name', required=True, type=str)
    parser.add_argument('--checkpoint', required=False, default=None, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--save_step', required=True, type=int)
    parser.add_argument('--warmup_step', required=True, type=int)
    parser.add_argument('--step_size', required=True, type=int)
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--max_seq_len', required=True, type=int)
    parser.add_argument('--n_hid_lyr', required=True, type=int)
    parser.add_argument('--n_epoch', required=True, type=int)
    parser.add_argument('--n_head', required=True, type=int)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--d_ff', required=True, type=int)
    parser.add_argument('--d_hid', required=True, type=int)
    parser.add_argument('--p_hid', required=True, type=float)
    parser.add_argument('--log_step', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--beta1', required=True, type=float)
    parser.add_argument('--beta2', required=True, type=float)
    parser.add_argument('--eps', required=True, type=float)
    # parser.add_argument('--max_norm', required=True, type=float)
    parser.add_argument('--wd', required=True, type=float)
    args = parser.parse_args()

    if args.step_size % args.batch_size != 0:
        raise ValueError('`step_size` must be divided by `batch_size`')

    # Save configuration.
    util.cfg.save(args)

    # Random seed initialization.
    util.seed.set_seed(seed=args.seed)

    # Load dataset and dataset config.
    dset = MLMDataset(exp_name=args.dataset_exp_name, n_sample=args.n_sample)
    dset_cfg = util.cfg.load(exp_name=args.dataset_exp_name)

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
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=len(os.sched_getaffinity(0)),
    )

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=dset_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr](exp_name=tknzr_cfg.exp_name)

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
    log_path = os.path.join(util.path.EXP_PATH, args.exp_name)
    writer = SummaryWriter(log_dir=log_path)

    # Log performance.
    pre_avg_loss = 0.0
    avg_loss = 0.0

    # Global optimization step.
    step = 0
    accumulation_steps = args.step_size / args.batch_size
    accumulation_count = 0
    start_epoch = 0

    # Load checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        avg_loss = checkpoint['loss']

    for epoch in range(start_epoch, args.n_epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {epoch}, loss: {avg_loss:.6f}',
        )
        for batch_mask_tkids, batch_target_tkids, batch_is_mask in tqdm_dldr:
            batch_mask_tkids = torch.LongTensor(batch_mask_tkids)
            batch_target_tkids = torch.LongTensor(batch_target_tkids)
            batch_is_mask = torch.FloatTensor(batch_is_mask)

            loss = model.loss_fn(
                batch_mask_tkids=batch_mask_tkids.to(device),
                batch_target_tkids=batch_target_tkids.to(device),
                batch_is_mask=batch_is_mask.to(device),
            )
            avg_loss += loss.item() / accumulation_steps

            loss.backward()

            accumulation_count += 1
            if accumulation_count == accumulation_steps:
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(),
                #     max_norm=args.max_norm,
                # )
                optim.step()
                optim.zero_grad()
                step += 1
                accumulation_count = 0

                if step <= args.warmup_step:
                    warmup_factor = min(1.0, step/args.warmup_step)
                    for parameters in optim.param_groups:
                        parameters['lr'] = args.lr*warmup_factor

                if step % args.save_step == 0:
                    model.save(
                        ckpt=step,
                        exp_name=args.exp_name,
                        loss=avg_loss,
                        optimizer=optim,
                        epoch=epoch,
                    )

                if step % args.log_step == 0:
                    avg_loss = avg_loss / args.log_step

                    tqdm_dldr.set_description(
                        f'epoch: {epoch}, loss: {avg_loss:.6f}'
                    )

                    # Log on tensorboard
                    writer.add_scalar(
                        f'loss',
                        avg_loss,
                        step,
                    )

                    # Refresh log performance.
                    # pre_avg_loss = avg_loss
                    avg_loss = 0.0

        # if pre_avg_loss < 0.5 and pre_avg_loss != 0.0:
        #     break
    # Save last checkpoint.
    model.save(
        ckpt=step,
        exp_name=args.exp_name,
        loss=avg_loss,
        optimizer=optim,
        epoch=epoch,
    )

    # Close tensorboard logger.
    writer.close()


if __name__ == '__main__':
    main()


"""
CUDA_VISIBLE_DEVICES=1 \
python train_mlm_model.py \
    --model transformer \
    --exp_name mlm_2M_12_lry \
    --dataset_exp_name mask_data_merged_2M \
    --batch_size 32 \
    --save_step 15000 \
    --warmup_step 10000 \
    --step_size 256 \
    --lr 1e-4 \
    --max_seq_len 400 \
    --n_hid_lyr 12 \
    --n_epoch 20 \
    --n_head 8 \
    --n_sample 4000000 \
    --d_ff 2048 \
    --d_hid 512 \
    --p_hid 0.1 \
    --log_step 100 \
    --seed 2022 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-08\
    --wd 0.01 
"""
