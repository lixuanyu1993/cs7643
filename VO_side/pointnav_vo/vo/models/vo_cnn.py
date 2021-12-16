#! /usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *


class ResNetEncoder(nn.Module):
    def __init__(
            self,
            *,
            observation_space,
            observation_size,
            baseplanes=32,
            ngroups=32,
            spatial_size_w=128,
            spatial_size_h=128,
            make_backbone=None,
            normalize_visual_inputs=False,
            after_compression_flat_size=2048,
            channels=[RGB_PAIR_CHANNEL, DEPTH_PAIR_CHANNEL, 0, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__()
        self.tags = [0, 0, 0, 0]
        self.tag_names = ["rgb", "depth", "discretized_depth", "top_down_view"]

        for i in range(self.tag_names):
            if self.tag_names[i] in observation_space:
                spatial_size_w, spatial_size_h = observation_size
                self.tags[i] = channels[i]
                if i == 2:
                    self.tags[i] = channels[i] * 2  # discretized_depth_channels * 2
        input_channels = sum(self.tags)

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)
        final_spatial_h = int(np.ceil(spatial_size_h * self.backbone.final_spatial_compress))
        final_spatial_w = int(np.ceil(spatial_size_w * self.backbone.final_spatial_compress))

        num_compression_channels = int(round(after_compression_flat_size / (final_spatial_w * final_spatial_h)))
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(inplace=True),
        )

        self.output_shape = (num_compression_channels, final_spatial_h, final_spatial_w,)

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def input_appender(self, cnn_input, ob, size, name):
        if size > 0:
            tmp = ob[name].permute(0, 3, 1, 2)
            cnn_input.append([tmp[:, : size // 2, :], tmp[:, size // 2:, :], ])

    def forward(self, observation_pairs):

        cnn_input = []
        for i in range(self.tag_names):
            self.input_appender(cnn_input, observation_pairs, tags[i], tag_names[i])

        cnn_input = [j for i in list(zip(*cnn_input)) for j in i]
        x = torch.cat(cnn_input, dim=1)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VO_Base(nn.Module):
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
            after_compression_flat_size=2048,
            channels=[RGB_PAIR_CHANNEL, DEPTH_PAIR_CHANNEL, 0, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            observation_space=observation_space,
            observation_size=observation_size,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            after_compression_flat_size=after_compression_flat_size,
            channels=channels
        )

        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output[1].weight)
        nn.init.constant_(self.output[1].bias, 0)

    def forward(self, observation_pairs):
        visual_feats = self.visual_encoder(observation_pairs)
        visual_feats = self.fc(visual_feats)
        output = self.output(visual_feats)
        return output


@baseline_registry.register_vo_model(name="vo_cnn")
class VO_CNN(VO_Base):
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
            channels=[0, 0, 0, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            channels=channels
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb")
class VO_CNNRGB(VO_Base):
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
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_wider")
class VO_CNNWider(VO_Base):
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
    ):
        # Make the encoder 2x wide will result in ~3x #params
        resnet_baseplanes = 2 * resnet_baseplanes

        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_deeper")
class VO_CNNDeeper(VO_Base):
    def __init__(
            self,
            *,
            observation_space,
            observation_size,
            hidden_size=512,
            resnet_baseplanes=32,
            backbone="resnet101",
            normalize_visual_inputs=False,
            output_dim=DEFAULT_DELTA_STATE_SIZE,
            dropout_p=0.2,
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_dd")
class VO_CNNDiscretizedDepth(VO_Base):
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
            channels=[0, 0, 10, 0]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            channels=channels,
            after_compression_flat_size=2048,
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_top_down")
class VO_CNN_RGB_D_TopDownView(VO_Base):
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
            channels=[0, 0, 10, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            channels=channels
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_dd_top_down")
class VO_CNN_RGB_DD_TopDownView(VO_Base):
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
            channels=[0, 0, 10, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            channels=channels,
        )


@baseline_registry.register_vo_model(name="vo_cnn_d_dd_top_down")
class VO_CNN_D_DD_TopDownView(VO_Base):
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
            channels=[0, 0, 10, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            channels=channels
        )


@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_dd_top_down")
class VO_CNNDiscretizedDepthTopDownView(VO_Base):
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
            channels=[0, 0, 10, TOP_DOWN_VIEW_PAIR_CHANNEL]
    ):
        super().__init__(
            observation_space=observation_space,
            observation_size=observation_size,
            hidden_size=hidden_size,
            resnet_baseplanes=resnet_baseplanes,
            backbone=backbone,
            normalize_visual_inputs=normalize_visual_inputs,
            output_dim=output_dim,
            dropout_p=dropout_p,
            after_compression_flat_size=2048,
            channels=channels
        )
