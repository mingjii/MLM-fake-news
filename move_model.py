import os
import util.cfg

file_name = "test_1000"
file_path = os.path.join("exp", file_name)
for exp in os.listdir(file_path):
    path = os.path.join(file_name, exp)
    cfg = util.cfg.load(path)
    if file_name not in cfg.exp_name:
        cfg.exp_name = path
        util.cfg.save(cfg)
    