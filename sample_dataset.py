"""
Sample dataset script.

Examples
========

.. code-block::

    python sample_dataset.py \
        --file_name news_v2.3.db \
        --dataset news \
        --idx 100
"""

import html
import argparse
from dataset import DSET_OPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_name',
        help='SQL file name to read.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--dataset',
        help='Which dataset to use.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--idx',
        action='extend',
        nargs='+',
        required=True,
        type=int,
    )
    args = parser.parse_args()

    dset = DSET_OPT[args.dataset](args.file_name, -1)

    titles = []
    articles = []
    # Remove repeated index, sort with ascending order.
    for idx in sorted(list(set(args.idx))):
        title, article = dset[idx]
        titles.append(title)
        articles.append(article)

    print('''
    <style>
        table {
            width: 100%;
            border: 1px solid black;
        }

        th, td {
            border: 1px solid black;
        }
    </style>
    ''')
    print('<table>')
    print('<tr><th>title</th><th>article</th></tr>')
    for title, article in zip(titles, articles):
        print(
            f'<tr><td>{html.escape(title)}</td><td>{html.escape(article)}</td></tr>'
        )
    print('</table>')


if __name__ == "__main__":
    main()
