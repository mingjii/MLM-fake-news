import os
import sqlite3

import torch

import util.path


class Seq2SeqNewsDataset(torch.utils.data.Dataset):
    r"""
    Dataset for seq2seq models.
    return a tuple contain title and article as below format
​
    return: (title, article)
    """
    dset_name = "news"

    def __init__(self, db_path: str, n_sample: int):
        super().__init__()
        # print(os.path.join(util.path.DATA_PATH, db_path))
        # exit()
        # Connect to DB.
        conn = sqlite3.connect(os.path.join(util.path.DATA_PATH, db_path))

        # Set `conn.row_factory` to get right return format.
        conn.row_factory = lambda cursor, row: row

        # Get database cursor.
        cursor = conn.cursor()

        self.titles = []
        self.articles = []

        # Get all news title and article.
        count = 0
        for title, article in iter(cursor.execute(
                'SELECT title, article from news_table;')):
            count += 1
            if count > n_sample and n_sample != -1:
                break
            self.titles.append(title)
            self.articles.append(article)

        conn.close()

    def __getitem__(self, index: int):
        return self.titles[index], self.articles[index]

    def __len__(self):
        return len(self.titles)
