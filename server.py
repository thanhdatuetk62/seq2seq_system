from bin import Controller

import os
import warnings
import yaml
import argparse

from flask import Flask

parser = argparse.ArgumentParser(description="Commandline arguments for \
    running seq2seq server.")

parser.add_argument("--config", type=str, default=None,
                    help="Path to config file in which contains hyperparameters")

parser.add_argument("--save_dir", type=str, default='.',
                    help="Path to saved model or want to save model")

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

class FlaskApp(Flask):
    def __init__(self, controller):
        # self.controller = controller
        self.controller = controller
        super().__init__(__name__)

controller = Controller(mode="infer", save_dir=args.save_dir, **options)
app = FlaskApp(controller=controller)

if __name__ == "__main__":
    # Suppress Warning
    warnings.filterwarnings("ignore")
    server_config = options.get("server_config", {})
    app.run(**server_config)
