from bin import Controller

import os
import argparse
import warnings
import yaml

import torch
import random
import numpy as np

torch.backends.cudnn.deterministic = True
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Commandline arguments for \
    running seq2seq system.")

parser.add_argument("mode", type=str, choices=["train", "infer"], \
    help="Choose type of using: [train] or [infer]")

parser.add_argument("--config", type=str, default=None, \
    help="Path to config file in which contains hyperparameters")

parser.add_argument("--save_dir", type=str, default='.', \
    help="Path to saved model or want to save model")

parser.add_argument("--checkpoint", type=int, default=None, \
    help="Load specified checkpoint of training")

parser.add_argument("--infer_src_path", type=str, \
    help="Path to src file which needed for inference")

parser.add_argument("--infer_save_path", type=str, default="output.txt", \
    help="Path to save file which is result of inference")


def main():
    # Parse arguments from standard input
    args = parser.parse_args()
    options = {}
    if args.config is not None:
        try:
            # Partially override default options by external option file
            options.update(yaml.load(open(args.config, 'r')))
        except:
            print("Cannot load options from file `{}` (YML format)."
                  .format(args.config))
    else:
        print("Using default config.")
    controller = Controller(mode=args.mode, save_dir=args.save_dir, **options)
    if args.mode == "train":
        train_config = options.get("train_config", {})
        controller.train(ckpt=args.checkpoint, train_config=train_config)
    if args.mode == "infer":
        infer_config = options.get("infer_config", {})
        controller.infer_from_file(src_path=args.infer_src_path,
                                   save_path=args.infer_save_path, 
                                   ckpt=args.checkpoint,
                                   **infer_config)


if __name__ == "__main__":
    # Suppress Warning
    warnings.filterwarnings("ignore")

    # Main
    main()
