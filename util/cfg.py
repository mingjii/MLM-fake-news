r"""Save training configuration and load pre-trained configuration."""

import argparse
import json
import os

import util.path


def save(args: argparse.Namespace) -> None:
    # Get file directory and path.
    file_dir = os.path.join(util.path.EXP_PATH, args.exp_name)
    file_path = os.path.join(file_dir, 'cfg.json')

    # Create experiment path if not exist.
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.isdir(file_path):
        raise FileExistsError(f'{file_path} is a directory.')

    # Save configuration in JSON format.
    with open(file_path, 'w', encoding='utf-8') as output_file:
        json.dump(args.__dict__, output_file, ensure_ascii=False)


def load(exp_name: str) -> argparse.Namespace:
    # Get file path.
    file_path = os.path.join(util.path.EXP_PATH, exp_name, 'cfg.json')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

    if os.path.isdir(file_path):
        raise FileExistsError(f'{file_path} is a directory.')

    # Load configuration from JSON file.
    with open(file_path, 'r', encoding='utf-8') as input_file:
        cfg = json.load(input_file)

    # Wrap configuration with `argparse.Namespace` for convenience.
    return argparse.Namespace(**cfg)
