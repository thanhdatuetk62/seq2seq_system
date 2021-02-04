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

parser.add_argument("mode", type=str, choices=["train", "infer", "compile"], \
    help="Choose type of using: [train], [infer], [compile]")

parser.add_argument("--config", type=str, default=None, \
    help="Path to config file in which contains hyperparameters")

parser.add_argument("--save_dir", type=str, default='.', \
    help="Path to saved model or want to save model")

parser.add_argument("--checkpoint", type=int, default=None, \
    help="Load specified checkpoint of training")

parser.add_argument("--src_path", type=str, \
    help="[infer] Path to src file which needed for inference")

parser.add_argument("--save_path", type=str, default="output.txt", \
    help="[infer] Path to save file which is result of inference")

parser.add_argument("--sos_token", type=str, default=None, \
    help="[infer] Start of sequence target token")

parser.add_argument("--export_path", type=str, default="export.pt", \
    help="[compile] Path to save TorchScript")


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
    controller = Controller(device=options.get("device", "cpu"), \
        save_dir=options.get("save_dir", "run"), \
        keep_checkpoints=options.get("keep_checkpoints", 100))
    if args.mode == "train":
        controller.train(ckpt=args.checkpoint, **options)
    if args.mode == "infer":
        controller.infer(src_path=args.src_path,
                         save_path=args.save_path, 
                         sos_token=args.sos_token,
                         ckpt=args.checkpoint, **options)
    if args.mode == "compile":
        infer_config = options.get("infer_config", {})
        controller.compile(ckpt=args.checkpoint, \
            export_path=args.export_path, **infer_config)


if __name__ == "__main__":
    # Suppress Warning
    warnings.filterwarnings("ignore")

    # Main
    main()
