"""
Sample dataset script.

Examples
========

.. code-block::

    python sample_with_pattern.py \
        --file_name news_v2.3.db \
        --dataset news \
        --n_sample 10 \
        --pattern "(?<\\!org)(?<\\!per)(?<\\!loc)(?<\\!\\d)\\d+>"
"""

import html
import re
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
        '--pattern',
        help='Searching pattern (can be a python regular expression).',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--n_sample',
        help='Max number of samples which containing given pattern.',
        required=True,
        type=int,
    )
    args = parser.parse_args()

    dset = DSET_OPT[args.dataset](args.file_name, -1)

    # Create matching pattern.
    pattern = re.compile(args.pattern)
    if args.n_sample == -1:
        args.n_sample = len(dset)

    titles = []
    articles = []
    count = 0
    stop_count = 0
    # Loop through dataset to find samples which containing given pattern.
    for title, article in iter(dset):
        stop_count += 1
        if pattern.search(title) or pattern.search(article):
            titles.append(title)
            articles.append(article)
            count += 1

            if count >= args.n_sample:
                break

    print('''
    <style>
        table {
            width: 100%;
            border: 1px solid black;
        }

        th, td {
            border: 1px solid black;
        }

        div {
            width: 100%;
        }
    </style>
    ''')
    print(f'<div>Number of matched samples: {len(titles)}</div>')
    print(f'<div>Search stop at index: {stop_count}</div>')
    print('<table>')
    print('<tr><th>title</th><th>article</th></tr>')
    for title, article in zip(titles, articles):
        print(
            f'<tr><td>{html.escape(title)}</td><td>{html.escape(article)}</td></tr>'
        )
    print('</table>')


if __name__ == "__main__":
    main()
