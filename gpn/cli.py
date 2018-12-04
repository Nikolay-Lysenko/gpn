"""
This is the single entry-point for all project-related tasks.

Author: Nikolay Lysenko
"""


import argparse
import os

import yaml

from gpn import discriminator_training, generator_training


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='GPNs')
    parser.add_argument(
        '-t', '--task', type=str, required=True,
        choices={'d_train', 'g_train', 'g_apply'},
        help='what to do: train discriminator or generator, apply generator'
    )
    parser.add_argument(
        '-d', '--dataset_name', type=str, required=True,
        choices={'mnist'},
        help='name of dataset to use'
    )
    parser.add_argument(
        '-c', '--config_path', type=str, default=None,
        help='path to config file in YAML'
    )
    cli_args = parser.parse_args()

    default_config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    cli_args.config_path = cli_args.config_path or default_config_path

    return cli_args


def main():
    """Run all necessary code."""
    cli_args = parse_cli_args()
    with open(cli_args.config_path) as config_file:
        all_settings = yaml.load(config_file)
    settings = all_settings[cli_args.dataset_name]

    if cli_args.task == 'd_train':
        discriminator_training.train(settings)
    elif cli_args.task == 'g_train':
        generator_training.train(settings)
    elif cli_args.task == 'g_apply':
        pass


if __name__ == '__main__':
    main()
