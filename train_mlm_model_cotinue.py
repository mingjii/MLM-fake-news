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
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--ckpt', required=True, type=int)
    parser.add_argument('--n_epoch', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()

    cfg = util.cfg.load(exp_name=args.exp_name)
    start_epoch = cfg.n_epoch
    cfg.seed = args.seed

    # Random seed initialization.
    util.seed.set_seed(seed=cfg.seed)

    # Load dataset and dataset config.
    dset = MLMDataset(exp_name=cfg.dataset_exp_name, n_sample=cfg.n_sample)
    dset_cfg = util.cfg.load(exp_name=cfg.dataset_exp_name)

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
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=dset_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr].load(exp_name=tknzr_cfg.exp_name)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Load model and global optimization step.
    model, step = MODEL_OPT[cfg.model].load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **cfg.__dict__,
    )
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
            'weight_decay': cfg.wd,
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
        betas=(cfg.beta1, cfg.beta2),
        lr=cfg.lr,
        eps=cfg.eps,
    )

    # Loggin.
    log_path = os.path.join(util.path.LOG_PATH, cfg.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    # Log performance.
    pre_avg_loss = 0.0
    avg_loss = 0.0
    step += step % cfg.log_step
    

    for epoch in range(start_epoch, args.n_epoch):
        tqdm_dldr = tqdm(
            dldr,
            desc=f'epoch: {epoch}, loss: {pre_avg_loss:.6f}',
        )
        cfg.n_epoch = epoch + 1
        for batch_mask_tkids, batch_target_tkids, batch_is_mask in tqdm_dldr:
            batch_mask_tkids = torch.LongTensor(batch_mask_tkids)
            batch_target_tkids = torch.LongTensor(batch_target_tkids)
            batch_is_mask = torch.FloatTensor(batch_is_mask)

            loss = model.loss_fn(
                batch_mask_tkids=batch_mask_tkids.to(device),
                batch_target_tkids=batch_target_tkids.to(device),
                batch_is_mask=batch_is_mask.to(device),
            )
            avg_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg.max_norm,
            )

            optim.step()
            optim.zero_grad()

            step += 1

            if step % cfg.ckpt_step == 0:
                model.save(ckpt=step, exp_name=cfg.exp_name)

            if step % cfg.log_step == 0:
                avg_loss = avg_loss / cfg.log_step

                # Log on tensorboard
                writer.add_scalar(
                    f'loss',
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
    model.save(ckpt=step, exp_name=cfg.exp_name)

    # Save configuration.
    util.cfg.save(cfg)

    # Close tensorboard logger.
    writer.close()


if __name__ == '__main__':
    main()


"""
CUDA_VISIBLE_DEVICES=1 python train_mlm_model_cotinue.py \
    --exp_name test_model \
    --ckpt -1 \
    --n_epoch 50 \
    --seed 32 
"""