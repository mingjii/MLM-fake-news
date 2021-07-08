import os

output_name = "exp3-3.html"
experiments = [
    ("test_1000/simple_test_mask10_pmask10_1000", 9000),
]
for i, (exp, ckpt) in enumerate(experiments):
    script = f"CUDA_VISIBLE_DEVICES=0 python eval_mlm_model.py \
    --ckpt {ckpt} \
    --exp_name {exp} \
    --k 1 \
    --seed 42 "
    if i == 0:
        script += " >> "+output_name
    else:
        script += " >> "+output_name
    os.system(script)