#! /usr/bin/env python


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_model.utils.misc_utils import Flatten
from habitat_model.utils.baseline_registry import baseline_registry
from habitat_model.model_utils.visual_encoders import resnet
from .vo_cnn import ResNetEncoder
from habitat_model.vo.common.common_vars import *


@baseline_registry.register_vo_model(name="vo_cnn_act_embed")
class VO_CNNActEmbed(nn.Module):
    def __init__(
            self,
            *,
            observation_space,
            observation_size,
            hidden_size=512,
            resnet_baseplanes=32,
            backbone="resnet18",
            normalize_visual_inputs=False,
            output_dim=DEFAULT_DELTA_STATE_SIZE,
            dropout_p=0.2,
            channels=[0, 0, 10, 0],
            after_compression_flat_size=2048,
            n_acts=N_ACTS,
    ):
        super().__init__()

        self.emb = nn.Embedding(n_acts + 1, EMBED_DIM)

        self.visual_encoder = ResNetEncoder(
            observation_space=observation_space,
            observation_size=observation_size,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            channels=channels,
            after_compression_flat_size=after_compression_flat_size,
        )

        self.flatten = Flatten()

        self.hidden = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(np.prod(self.visual_encoder.output_shape) + EMBED_DIM, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs, actions):
        tmp = self.visual_encoder(observation_pairs)
        tmp = self.flatten(tmp)
        tmp = self.hidden(torch.cat((tmp, self.emb(actions)), dim=1))
        tmp = self.output(tmp)
        return tmp


@baseline_registry.register_vo_model(name="vo_cnn_wider_act_embed")
class VO_CNNWiderActEmbed(VO_CNNActEmbed):
    def __init__(
            self,
            *,
            observation_space,
            observation_size,
            hidden_size=512,
            resnet_baseplanes=32,
            backbone="resnet18",
            normalize_visual_inputs=False,
            output_dim=DEFAULT_DELTA_STATE_SIZE,
            dropout_p=0.2,
            n_acts=N_ACTS,
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=2 * resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            n_acts=n_acts,
        )
