import os

experiments = [
    (10000, 10, 15),
    (10000, 10, 10),
    (10000, 10, 5),
    (10000, 7, 15),
    (10000, 7, 10),
    (10000, 7, 5),
    (10000, 4, 15),
    (10000, 4, 10),
    (10000, 4, 5),
]
for n_sample, n_mask, p_mask in experiments:
    data_size = n_sample//10000
    data_path = f'data_{data_size}w'
    path = os.path.join(data_path, f'mask_data_len512_pmask{p_mask}_mlen{n_mask}_n5_{data_size}w')
    os.system(f"python create_mlm_dataset.py \
    --exp_name '{path}' \
    --dataset news \
    --file_name 'news_v2.3.db' \
    --max_seq_len 512 \
    --n_epoch 5 \
    --n_sample {n_sample} \
    --seed 42 \
    --p_mask {p_mask/100} \
    --p_len 0.2 \
    --max_span_len {n_mask} \
    --tknzr_exp_name 'tknzr_news_min6_v2.3'")