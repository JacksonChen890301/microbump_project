import argparse
import torch
import numpy as np
from numpy import *


class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, inputs, output=1, method="transformer_attribution", mode=0, start_layer=0):
        pred = self.model(inputs)
        kwargs = {"alpha": 1}
        pred = pred.requires_grad_(True)
        self.model.zero_grad()
        pred.backward(retain_graph=True)
        return self.model.relprop(torch.tensor(output).to(inputs.device), method=method, mode=mode,
                                  start_layer=start_layer, **kwargs)

