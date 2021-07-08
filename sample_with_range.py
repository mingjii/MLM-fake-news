import os


script = '''python sample_dataset_mlm.py \
    --file_name mask_data_len512_pmask5_sample1k \
    --dataset mlm \
'''

for i in range(0, 10):
    script += f'--idx {i} '

script += '> tmp_p5_mask10.html'

os.system(script)
