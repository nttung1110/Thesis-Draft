import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def apply_sigmoid(value):
    return 1/(1+np.exp(-value))

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNet, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x