import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import math

class ConcatModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(ConcatModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        x = self.modelB(self.modelA(x))
        return x 
    