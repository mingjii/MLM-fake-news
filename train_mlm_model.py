import argparse
import random
import torch
import torch.optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

import util.path
import util.cfg
import util.dist
import util.seed
from model import MODEL_OPT
# from dataset import MLMDataset
# from dataset import NewsDataset
from dataset import DSET_OPT
from tokenizer import TKNZR_OPT


def main():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--dataset_type', required=True, type=str)
    parser.add_argument('--dataset_exp_name', required=True, type=str)
    parser.add_argument('--tknzr_exp_name', required=True, type=str)
    parser.add_argument('--p_mask', required=False, default=0.15, type=float)
    parser.add_argument('--p_len', required=False, default=0.1, type=float)
    parser.add_argument('--checkpoint', required=False, default=None, type=str)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--save_step', required=True, type=int)
    parser.add_argument('--warmup_step', required=True, type=int)
    parser.add_argument('--step_size', required=True, type=int)
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--n_hid_lyr', required=True, type=int)
    parser.add_argument('--n_epoch', required=True, type=int)
    parser.add_argument('--n_head', required=True, type=int)
    parser.add_argument('--n_sample', required=True, type=int)
    parser.add_argument('--d_ff', required=True, type=int)
    parser.add_argument('--d_hid', required=True, type=int)
    parser.add_argument('--p_hid', required=True, type=float)
    parser.add_argument('--max_seq_len', required=True, type=int)
    parser.add_argument('--max_span_len', required=False, default=9, type=int)
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

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=args.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr](exp_name=tknzr_cfg.exp_name)
    sp_tkids = [tknzr.cls_tkid, tknzr.sep_tkid, tknzr.pad_tkid]

    # Load dataset and dataset config.
    print("load data....")
    # dset = MLMDataset(exp_name=args.dataset_exp_name, n_sample=args.n_sample)
    dset = DSET_OPT[args.dataset_type](args.dataset_exp_name, args.n_sample)
    # dset_cfg = util.cfg.load(exp_name=args.dataset_exp_name)

    def collate_fn(batch):
        batch_mask_tkids = []
        batch_target_tkids = []
        batch_is_mask = []
        for title, article in batch:
            target_tkids = tknzr.enc(
                txt=title,
                txt_pair=article,
                max_seq_len=args.max_seq_len
            )

            mask_tkids = []
            is_mask = []
            mask_span_count = 0
            while sum(is_mask) == 0:
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
            batch_is_mask.append(is_mask)
            batch_mask_tkids.append(mask_tkids)
            batch_target_tkids.append(target_tkids)

        return batch_mask_tkids, batch_target_tkids, batch_is_mask

    # Create data loader.
    print("initialize dataloader....")
    dldr = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=len(os.sched_getaffinity(0)),
    )
    # return dldr
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load model.
    print("initialize model....")
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda x: min(1.0, x/args.warmup_step)
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
        print("load checkpoint model....")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        avg_loss = checkpoint['loss']

    for epoch in range(start_epoch, args.n_epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {epoch}, loss: {avg_loss:.6f}',
        )
        for batch_mask_tkids, batch_target_tkids, batch_is_mask in tqdm_dldr:
            batch_mask_tkids = torch.LongTensor(batch_mask_tkids).to(device)
            batch_target_tkids = torch.LongTensor(batch_target_tkids).to(device)
            batch_is_mask = torch.FloatTensor(batch_is_mask).to(device)
            
            loss = model.loss_fn(
                batch_mask_tkids=batch_mask_tkids,
                batch_target_tkids=batch_target_tkids,
                batch_is_mask=batch_is_mask,
            )
            # tqdm_dldr.set_description(
            #             f'epoch: {epoch}, loss: {loss.item():.6f}'
            #         )
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
                scheduler.step()
                step += 1
                accumulation_count = 0

                # if step <= args.warmup_step:
                #     warmup_factor = min(1.0, step/args.warmup_step)
                #     for parameters in optim.param_groups:
                #         parameters['lr'] = args.lr*warmup_factor

                if step % args.save_step == 0:
                    model.save(
                        ckpt=step,
                        exp_name=args.exp_name,
                        loss=avg_loss,
                        optimizer=optim,
                        scheduler=scheduler,
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
        scheduler=scheduler,
        epoch=epoch,
    )

    # Close tensorboard logger.
    writer.close()


if __name__ == '__main__':
    main()


"""
CUDA_VISIBLE_DEVICES=1 \
python -i train_mlm_model.py \
    --model transformer \
    --exp_name mlm_2M_l12 \
    --dataset_type news \
    --dataset_exp_name merged_cna_ettoday_storm.db \
    --tknzr_exp_name tknzr_sentPiece_3w5 \
    --batch_size 16 \
    --save_step 10000 \
    --warmup_step 10000 \
    --step_size 128 \
    --lr 1e-4 \
    --n_hid_lyr 12 \
    --n_epoch 20 \
    --n_head 8 \
    --n_sample -1 \
    --max_seq_len 400 \
    --d_ff 1024 \
    --d_hid 512 \
    --p_hid 0.1 \
    --log_step 10 \
    --seed 2022 \
    --beta1 0.9 \
    --beta2 0.999 \
    --eps 1e-08\
    --wd 0.01 
"""
