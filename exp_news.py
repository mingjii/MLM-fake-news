import os


exp_pairs = [
    ("mask_data_n10_m7_p10_v2.3", "n10_m7_p10_v2.3"),
    ]
for data_name, exp_name in exp_pairs:
    os.system(f'python train_mlm_model.py \
        --batch_size 168 \
        --beta1 0.9 \
        --beta2 0.999 \
        --ckpt_step 100000 \
        --d_ff 2048 \
        --d_hid 768 \
        --dataset_exp_name "{data_name}" \
        --eps 1e-8 \
        --exp_name "{exp_name}" \
        --log_step 200 \
        --lr 1e-4 \
        --max_norm 3.0 \
        --max_seq_len 512 \
        --model transformer \
        --n_epoch 50 \
        --n_head 8 \
        --n_hid_lyr 6 \
        --n_sample -1 \
        --p_hid 0.0 \
        --seed 40 \
        --wd 1e-1'
            )




