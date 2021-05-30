import argparse

from utils import set_seed, configure_model
from models import Avatar_Generator_Model
import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "avatar_image_generator"


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-nw  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-n', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-nt', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

    args = ap.parse_args()

    return args


def train(config_file, use_wandb, run_name, run_notes):
    set_seed(32)
    config = configure_model(config_file, use_wandb)

    if use_wandb:
        wandb.init(project=PROJECT_WANDB, config=config, name=run_name, notes=run_notes)
        config = wandb.config
        wandb.watch_called = False

    xgan = Avatar_Generator_Model(config, use_wandb)
    xgan.train()


if __name__ == '__main__':
    args = parse_arguments()

    use_wandb = args.wandb
    run_name = args.run_name
    run_notes = args.run_notes

    train(CONFIG_FILENAME, use_wandb=use_wandb, run_name=run_name, run_notes=run_notes)
