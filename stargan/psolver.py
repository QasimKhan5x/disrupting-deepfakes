import torch
import torch.nn as nn
from solver import Solver

from detection.dummy import DummyDetector2D
from loss import DisruptionLoss
from perturbation.unet2d import UNet2D


class Disruptor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.G = Solver(None, None, config).G
        self.D = DummyDetector2D()
        self.P = UNet2D(n_channels=3, n_classes=3)
        
        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.D.parameters():
            param.requires_grad = False
        
        self.criterion = DisruptionLoss(config)
        self.config = config
        
        
    def train(self):
        pass
        
