import os
import sqlite3
import pickle
import torch

import util.path


class MLMDataset(torch.utils.data.Dataset):
    r"""
    Dataset for MLM models.

    Return a tuple contain mask token ids, target token ids and whether
    token ids were masked.
​
    return: (mask_tkids, target_tkids, is_mask)
    """
    dset_name = 'mlm'
    file_name = 'mlm_dataset.db'

    def __init__(self, exp_name: str, n_sample: int):
        super().__init__()

        # Connect to DB.
        conn = sqlite3.connect(os.path.join(
            util.path.EXP_PATH,
            exp_name,
            self.__class__.file_name
        ))

        # Get database cursor.
        cursor = conn.cursor()

        self.all_mask_tkids = []
        self.all_target_tkids = []
        self.all_is_mask = []

        # Get all news title and article.
        count = 0
        for b_mask_tkids, b_target_tkids, b_is_mask in iter(cursor.execute(
                'SELECT mask_tkids, target_tkids, is_mask from mlm;')):
            count += 1
            if count > n_sample and n_sample != -1:
                break
            self.all_mask_tkids.append(pickle.loads(b_mask_tkids))
            self.all_target_tkids.append(pickle.loads(b_target_tkids))
            self.all_is_mask.append(pickle.loads(b_is_mask))

        conn.close()

    def __getitem__(self, idx: int):
        return (
            self.all_mask_tkids[idx],
            self.all_target_tkids[idx],
            self.all_is_mask[idx],
        )

    def __len__(self):
        return len(self.all_mask_tkids)
