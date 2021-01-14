from flask import Flask
from bin.data import DataController
import torch

class FlaskApp(Flask):
    def __init__(self, model_path, save_dir, device="cpu"):
        # self.controller = controller
        self.model_path = model_path
        self.save_dir = save_dir
        self.status = 1
        self.device = device
        self.batch_size = 32
        self.data = DataController(save_dir=save_dir, train=False, device=device)

        # Test
        self.wakeup()
        super().__init__(__name__)

    def wakeup(self):
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

    def sleep(self):
        self.model = None
