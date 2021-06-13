import os


script = '''python sample_dataset.py \
    --file_name news_revise.db \
    --dataset news \
'''

for i in range(0, 10):
    script += f'--idx {i} '

script += '> tmp_raw.html'

os.system(script)
