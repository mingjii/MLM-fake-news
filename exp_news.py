import os

# os.system(f'CUDA_VISIBLE_DEVICES=1 python train_mlm_model.py \
#     --batch_size 20 \
#     --beta1 0.9 \
#     --beta2 0.999 \
#     --ckpt_step 10000 \
#     --d_ff 2048 \
#     --d_hid 768 \
#     --dataset_exp_name "mask_data_len512_pmask5_mlen7_n5_2w" \
#     --eps 1e-8 \
#     --exp_name test_p5_l7_n5_10w \
#     --log_step 50 \
#     --lr 5e-5 \
#     --max_norm 5.0 \
#     --max_seq_len 512 \
#     --model transformer \
#     --n_epoch 50 \
#     --n_head 8 \
#     --n_hid_lyr 6 \
#     --n_sample -1 \
#     --p_hid 0.0 \
#     --seed 42 \
#     --wd 1e-2'
#           )


exp_pairs = [
    ("data_1k/mask_data_len512_pmask10_mlen7_sample1k",
     "data_1k/simple_test_mask7_pmask10_1k"),
    # ("data_1w/mask_data_len512_pmask15_mlen7_n5_1w", "simple_test_mask7_pmask15_1w"),
    # ("data_1w/mask_data_len512_pmask15_mlen4_n5_1w", "simple_test_mask4_pmask15_1w"),
    # ("data_1w/mask_data_len512_pmask10_mlen10_n5_1w",
    #  "simple_test_mask10_pmask10_1w"),
    # ("data_1w/mask_data_len512_pmask10_mlen7_n5_1w", "simple_test_mask7_pmask10_1w"),
    # ("data_1w/mask_data_len512_pmask10_mlen4_n5_1w", "simple_test_mask4_pmask10_1w"),
    # ("data_1w/mask_data_len512_pmask5_mlen10_n5_1w", "simple_test_mask10_pmask5_1w"),
    # ("data_1w/mask_data_len512_pmask5_mlen7_n5_1w", "simple_test_mask7_pmask5_1w"),
    # ("data_1w/mask_data_len512_pmask5_mlen4_n5_1w", "simple_test_mask4_pmask5_1w"),
]
for data_name, exp_name in exp_pairs:
    os.system(f'python train_mlm_model.py \
        --batch_size 20 \
        --beta1 0.9 \
        --beta2 0.99 \
        --ckpt_step 1000 \
        --d_ff 2048 \
        --d_hid 768 \
        --dataset_exp_name "{data_name}" \
        --eps 1e-8 \
        --exp_name "{exp_name}" \
        --log_step 10 \
        --lr 5e-5 \
        --max_norm 3.0 \
        --max_seq_len 512 \
        --model transformer \
        --n_epoch 70 \
        --n_head 8 \
        --n_hid_lyr 3 \
        --n_sample -1 \
        --p_hid 0.0 \
        --seed 12 \
        --wd 1e-2'
              )
