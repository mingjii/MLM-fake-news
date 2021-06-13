import os

data_version = "v2.2"
model = "Transformer"
layer = 3
tknzr_list = ["char", "bert_base_chinese"]

for tknzr in tknzr_list:
    tknzr_exp_name = f"tknzr_news_{data_version}"
    if tknzr != "char":
        tknzr_exp_name = f"pretknzr_news_{data_version}"

    os.system(f"python train_mlm_model.py \
        --model {model} \
        --data news_{data_version}.db \
        --exp_name {tknzr}/{model}_{tknzr_exp_name}_layer{layer} \
        --tknzr_exp_name {tknzr_exp_name} \
        --tknzr_type {tknzr} \
        --dset news \
        --n_epoch 20 \
        --batch_size 32 \
        --lr 5e-4 \
        --ckpt_step 10000 \
        --log_step 50 \
        --max_seq_len 512 \
        --d_emb 512 \
        --d_hid 768 \
        --n_hid_lyr {layer} \
        --p_emb 0.0 \
        --p_hid 0.0 \
        --seed 42 \
        --max_norm 3.0 \
        --eps 1e-8 \
        --beta1 0.9 \
        --beta2 0.99 \
        --wd 1e-2"
              )
    exit()
