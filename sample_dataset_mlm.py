"""
Sample dataset script.

Examples
========

.. code-block::

    python sample_dataset.py \
        --file_name simple_test_100 \
        --dataset mlm \
        --idx 100
"""

import html
import argparse
from dataset.mlm import MLMDataset
import util.cfg
from tokenizer import TKNZR_OPT


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

    dset = MLMDataset(exp_name=args.file_name, n_sample=-1)
    dset_cfg = util.cfg.load(exp_name=args.file_name)

    # Load tokenizer and config.
    tknzr_cfg = util.cfg.load(exp_name=dset_cfg.tknzr_exp_name)
    tknzr = TKNZR_OPT[tknzr_cfg.tknzr].load(exp_name=tknzr_cfg.exp_name)

    mask_data = []
    target_data = []
    # Remove repeated index, sort with ascending order.
    for idx in sorted(list(set(args.idx))):
        mask_tkids, target_tkids, is_mask = dset[idx]
        mask_data.append(tknzr.dec(mask_tkids))
        target_data.append(tknzr.dec(target_tkids))

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
    print('<tr><th>mask</th><th>target</th></tr>')
    for mask, target in zip(mask_data, target_data):
        print(
            f'<tr><td>{html.escape(mask)}</td><td>{html.escape(target)}</td></tr>'
        )
    print('</table>')


if __name__ == "__main__":
    main()
