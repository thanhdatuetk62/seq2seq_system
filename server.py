from bin import Controller
import torch
import os
import warnings
import yaml
import argparse

from serving import FlaskApp

parser = argparse.ArgumentParser(description="Commandline arguments for \
    running seq2seq server.")

parser.add_argument("--config", type=str, default=None,
                    help="Path to config file in which contains hyperparameters")

parser.add_argument("--save_dir", type=str, default=None,
                    help="Path to saved data utils")

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

app = FlaskApp(model_path=options.get("model_path"), \
    save_dir=args.save_dir, device=options.get("device", "cpu"))
# Load all APIs
from serving.translate import *

if __name__ == "__main__":
    # Suppress Warning
    warnings.filterwarnings("ignore")
    server_config = options.get("server_config", {})
    app.run(**server_config)
