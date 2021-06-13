import argparse
from tokenizer import TKNZR_OPT
from dataset import DSET_OPT
import util.cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_name',
        help='Dataset experiment name.',
        required=True,
        type=str
    )
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
        '--tknzr_exp_name',
        help='Tokenizer experiment name.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--n_sample',
        required=True,
        type=int
    )
    parser.add_argument(
        '--seed',
        action='extend',
        help='Different random seeds to generate dataset.',
        narg='+',
        required=True,
        type=int
    )
    args = parser.parse_args()

    cfg = util.cfg.load(exp_name=args.tknzr_exp_name)
    tknzr = TKNZR_OPT[cfg.tknzr].load(exp_name=args.tknzr_exp_name)

    # Load dataset.
    dset = DSET_OPT[args.dataset](args.file_name, args.n_sample)

    for seed in args.seed:
        pass


if __name__ == '__main__':
    main()


"""
python infer_tokenizer.py \
    --exp_name tknzr_news_v2.3 \
    --txt "你好嗎" \
    --txt_pair "<per0><org2><mask>"
"""
