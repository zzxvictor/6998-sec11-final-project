import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import matplotlib.pyplot as plt
import torchvision.utils
from functools import reduce
from operator import __add__

def compute_padding(k):
    return (k - 1) // 2

class TorchDetector(nn.Module):
    def __init__(self):
        super(TorchDetector, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=compute_padding(7)),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=compute_padding(3)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=compute_padding(3)),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=compute_padding(3)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=compute_padding(3)),
            nn.MaxPool2d(2, stride=2)
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(92416, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass
        output = self.cnn1(x)
        # output = torch.reshape(output, (output.size(0), -1))
        output = self.fc1(output)
        return output