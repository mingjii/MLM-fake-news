import os


os.system(f'python train_mlm_model.py \
    --batch_size 32 \
    --beta1 0.9 \
    --beta2 0.99 \
    --ckpt_step 10000 \
    --d_ff 256 \
    --d_hid 128 \
    --dataset_exp_name "1st_data" \
    --eps 1e-8 \
    --exp_name test_model \
    --log_step 5000 \
    --lr 5e-4 \
    --max_norm 3.0 \
    --max_seq_len 512 \
    --model transformer \
    --n_epoch 1 \
    --n_head 8 \
    --n_hid_lyr 2 \
    --n_sample 10 \
    --p_hid 0.1 \
    --seed 42 \
    --wd 1e-2'
          )
