import argparse

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

parser.add_argument("--infer_save_path", type=str, \
    help="Path to save file which is result of inference")
