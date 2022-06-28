import numpy as np
import torch
from torch import nn

def create_mlp(input_size, hidden_size, output_size, num_layer=3):
    
    sizes = [input_size] + [hidden_size] * (num_layer - 1) + [output_size]
    layers = []
    for i in range(num_layer):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i != num_layer - 1:
            layers.append(nn.ReLU())

    return layers

class Diagonal(nn.Module):

    def forward(self, x):
        B, C, C1 = x.shape
        assert C == C1

        index0 = torch.arange(B)
        index0 = torch.t(index0.repeat([C, 1])) # B, C
        index1 = torch.arange(C)
        index1 = index1.repeat([B, 1]) # B, C
        output = x[index0, index1, index1] # B, C

        return output

def create_thresholder(cfg, input_size, num_class):
    layers = []
    layers.extend(create_mlp(input_size, cfg.hidden_size, num_class, num_layer=cfg.num_layer))
    layers.append(Diagonal())
    thresholder = nn.Sequential(*layers)
    return thresholder

