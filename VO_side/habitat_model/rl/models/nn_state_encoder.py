#!/usr/bin/env python3

import torch
import torch.nn as nn

class NNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int
    ):
        super().__init__()

        # just for capatbility
        self.num_recurrent_layers = 1

        self.nn_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        for name, param in self.nn_model.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)


    def forward(self, x) -> torch.Tensor:
        y = self.nn_model(x)

        y = y.squeeze(0)
        return y