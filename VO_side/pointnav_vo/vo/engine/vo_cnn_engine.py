#! /usr/bin/env python

import os
import contextlib
import joblib
import random
import numpy as np
from tqdm import tqdm
from collections import deque, defaultdict, OrderedDict

import torch

import habitat
from habitat import Config, logger

from pointnav_vo.utils.config_utils import update_config_log
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.vo.common.common_vars import *


@baseline_registry.register_vo_engine(name="vo_cnn_base_enginer")
class VO_BaseEngine:
    def __init__(self, config=None, run_type="train", verbose=True):
        self._run_type = run_type
        self._pin_memory_flag = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self._config = config
        self._verbose = verbose

        if run_type == "train" and self._config.RESUME_TRAIN:
            self._config = torch.load(self._config.RESUME_STATE_FILE)["config"]
            self._config.defrost()
            self._config.RESUME_STATE_FILE = config.RESUME_STATE_FILE
            self._config.VO.TRAIN.epochs = config.VO.TRAIN.epochs
            self._config.RESUME_TRAIN = config.RESUME_TRAIN
            self._config = update_config_log(self._config, run_type, config.LOG_DIR)
            self._config.freeze()

        if "eval" in run_type:
            assert os.path.isfile(self._config.EVAL.EVAL_CKPT_PATH)
            self._config = torch.load(self._config.EVAL.EVAL_CKPT_PATH)["config"]
            self._config.defrost()
            self._config.RESUME_TRAIN = False
            self._config.EVAL = config.EVAL
            self._config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN
            self._config.VO.DATASET.EVAL = config.VO.DATASET.EVAL
            self._config.VO.EVAL.save_pred = config.VO.EVAL.save_pred
            self._config = update_config_log(self._config, run_type, config.LOG_DIR)
            self._config.freeze()

        self._config.defrost()

        if "PARTIAL_DATA_N_SPLITS" not in self._config.VO.DATASET:
            self._config.VO.DATASET.PARTIAL_DATA_N_SPLITS = 1

        self._config.freeze()
        if self.verbose:
            logger.info(f"Visual Odometry configs:\n{self._config}")

        self.flush_secs = 30
        self._observation_space = self.config.VO.MODEL.visual_type
        self._act_type = self.config.VO.TRAIN.action_type
        if isinstance(self._act_type, int):
            self._act_list = [self._act_type]
        elif isinstance(self._act_type, list):
            self._act_list = [TURN_LEFT, TURN_RIGHT]

        self.train_loader = None
        self.eval_loader = None
        self.separate_eval_loaders = None

        self._set_up_dataloader()
        self._set_up_model()
        if self._run_type == "train":
            self._set_up_optimizer()

    @property
    def config(self):
        return self._config

    @property
    def verbose(self):
        return self._verbose

    def _set_up_model(self):
        raise NotImplementedError

    def _set_up_dataloader(self):
        raise NotImplementedError

    def _save_ckpt(self, epoch):
        raise NotImplementedError

    def _set_up_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def _compute_loss(self, pred_delta_states, target_delta_states, d_type="dx", loss_weights=DEFAULT_LOSS_WEIGHTS,
                      dz_regress_masks=None, ):
        delta_xs, delta_zs, delta_yaws = target_delta_states
        delta = {"dx": delta_xs, "dz": delta_zs, "dyaw": delta_yaws}
        if d_type == "dx" or d_type == "dyaw":
            tmp_d = delta[d_type]
            d_type_diffs = (tmp_d - pred_delta_states) ** 2
            d_type_loss = torch.mean(d_type_diffs * loss_weights[d_type])
            target_magnitude = torch.mean(torch.abs(tmp_d)) + EPSILON
            abs_diff = torch.mean(torch.sqrt(d_type_diffs.detach()))
            return d_type_loss, abs_diff, target_magnitude, abs_diff / target_magnitude
        elif d_type == "dz":
            delta_z_diffs = (delta_zs - pred_delta_states) ** 2
            if dz_regress_masks is not None:
                delta_z_diffs = dz_regress_masks * delta_z_diffs
                filtered_dz_idxes = torch.nonzero(dz_regress_masks == 1.0, as_tuple=True)[0]
            else:
                filtered_dz_idxes = torch.tensor(np.arange(delta_zs.size()[0]))

            loss_dz = torch.mean(delta_z_diffs * loss_weights["dz"])
            if filtered_dz_idxes.size(0) == 0:
                target_magnitude_dz = torch.zeros(1) + EPSILON
                abs_diff_dz = torch.zeros(1)
            else:
                target_magnitude_dz = (torch.mean(torch.abs(delta_zs[filtered_dz_idxes])) + EPSILON)
                abs_diff_dz = torch.mean(torch.sqrt(delta_z_diffs.detach()[filtered_dz_idxes]))
            return loss_dz, abs_diff_dz, target_magnitude_dz, abs_diff_dz / target_magnitude_dz

    def _compute_loss_weights(self, actions, dxs, dys, dyaws):
        if "loss_weight_fixed" in self.config.VO.TRAIN and self.config.VO.TRAIN.loss_weight_fixed:
            loss_weights = {
                k: torch.ones(dxs.size()).to(dxs.device) * v
                for k, v in self.config.VO.TRAIN.loss_weight_multiplier.items()
            }
        else:
            no_noise_ds = np.array([NO_NOISE_DELTAS[int(_)] for _ in actions])
            no_noise_ds = torch.from_numpy(no_noise_ds).float().to(dxs.device)

            loss_weights = {}
            multiplier = self.config.VO.TRAIN.loss_weight_multiplier
            d_list = ["dx", "dz", "dyaw"]
            for i in range(len(d_list)):
                td = d_list[i]
                loss_weights[td] = torch.exp(multiplier[td] * torch.abs(no_noise_ds[:, i].unsqueeze(1) - dxs))

            for v in loss_weights.values():
                torch.all(v >= 1.0)

        return loss_weights

    def _log_grad(self, writer, global_step, grad_info_dict, d_type="dx"):

        if d_type not in grad_info_dict:
            grad_info_dict[d_type] = {}

        if isinstance(self.vo_model, dict):
            for k in self.vo_model:
                for n, p in self.vo_model[k].named_parameters():
                    if p.requires_grad:
                        writer.add_histogram(
                            f"{d_type}-Grad/{k}-{n}", p.grad.abs(), global_step
                        )
                        if f"{k}-{n}" not in grad_info_dict[d_type]:
                            grad_info_dict[d_type][f"{k}-{n}"] = []
                        grad_info_dict[d_type][f"{k}-{n}"].append(
                            p.grad.abs().mean().item()
                        )
        else:
            for n, p in self.vo_model.named_parameters():
                if p.requires_grad:
                    writer.add_histogram(
                        f"{d_type}-Grad/{n}", p.grad.abs(), global_step
                    )
                    if n not in grad_info_dict[d_type]:
                        grad_info_dict[d_type][n] = []
                    grad_info_dict[d_type][n].append(p.grad.abs().mean().item())

        if global_step > 0 and global_step % self.config.LOG_INTERVAL == 0:
            self._save_dict(
                grad_info_dict[d_type],
                os.path.join(self.config.LOG_DIR, f"avg_abs_grad_model_{d_type}.p"),
            )
            grad_info_dict[d_type] = {}

    def _regress_log_func(self, writer, split, global_step, abs_diff, target_magnitude, relative_diff, d_type="dx", ):
        writer.add_scalar(f"{split}_regression/{d_type}_abs_diff", abs_diff, global_step=global_step)
        writer.add_scalar(f"{split}_regression/{d_type}_target_magnitude", target_magnitude, global_step=global_step, )
        writer.add_scalar(f"{split}_regression/{d_type}_relative_diff", relative_diff, global_step=global_step, )

    def _regress_udpate_dict(self, split, abs_diffs, target_magnitudes, relative_diffs, d_type="dx"):

        info_dict = defaultdict(list)
        info_dict[f"abs_diff_{d_type}"].append(abs_diffs.item())
        info_dict[f"target_{d_type}_magnitude"].append(target_magnitudes.item())
        info_dict[f"relative_diff_{d_type}"].append(relative_diffs.item())

        save_f = os.path.join(self.config.INFO_DIR, f"{split}_regression_info.p")
        self._save_dict(dict(info_dict), save_f)

    def _save_dict(self, save_dict, f_path):
        if not os.path.isfile(f_path):
            tmp_dict = save_dict
        else:
            with open(f_path, "rb") as f:
                tmp_dict = joblib.load(f)
                for k, v in save_dict.items():
                    if k in tmp_dict:
                        tmp_dict[k].extend(v)
                    else:
                        tmp_dict[k] = v
        with open(f_path, "wb") as f:
            joblib.dump(tmp_dict, f, compress="lz4")
