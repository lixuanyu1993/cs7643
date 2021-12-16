#! /usr/bin/env python

import os
import contextlib
import joblib
import random
import time
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import sys
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch import autograd

import habitat
from habitat import Config, logger

from habitat_model.utils.tensorboard_utils import TensorboardWriter
from habitat_model.utils.baseline_registry import baseline_registry
from habitat_model.vo.dataset.regression_geo_invariance_iter_dataset import (StatePairRegressionDataset,
                                                                           normal_collate_func, fast_collate_func, )
from habitat_model.vo.engine.vo_cnn_engine import VO_BaseEngine
from habitat_model.vo.common.common_vars import *

TRAIN_NUM_WORKERS = 4
EVAL_NUM_WORKERS = 4
PREFETCH_FACTOR = 2
TIMEOUT = 5 * 60

DELTA_DIM = 3


@baseline_registry.register_vo_engine(name="vo_cnn_regression_geo_invariance_engine")
class VO_CNNRegressionGeometricInvarianceEngine(VO_BaseEngine):
    @property
    def delta_types(self):
        return DEFAULT_DELTA_TYPES

    @property
    def geo_invariance_types(self):
        return self.config.VO.GEOMETRY.invariance_types

    def _set_up_model(self):

        vo_model_cls = baseline_registry.get_vo_model(self.config.VO.MODEL.name)
        self.vo_model = {}
        for act in self._act_list:
            obs_size = (self.config.VO.VIS_SIZE_W, self.config.VO.VIS_SIZE_H)

            top_down_view_pair_channel = 0
            if "top_down_view" in self._observation_space:
                top_down_view_pair_channel = 2

            self.vo_model[act] = vo_model_cls(
                observation_space=self._observation_space,
                observation_size=obs_size,
                hidden_size=self.config.VO.MODEL.hidden_size,
                backbone=self.config.VO.MODEL.visual_backbone,
                normalize_visual_inputs=True,
                output_dim=DELTA_DIM,
                dropout_p=self.config.VO.MODEL.dropout_p,
                channels=[0, 0, self._discretized_depth_channels, top_down_view_pair_channel]
            )
            self.vo_model[act].to(self.device)

        if self._run_type == "train":
            if self.config.VO.MODEL.pretrained:
                for act in self._act_list:
                    act_str = ACT_IDX2NAME[act]
                    logger.info(f"Initializing {act_str} model from {self.config.VO.MODEL.pretrained_ckpt[act_str]}")
                    ckpt = torch.load(self.config.VO.MODEL.pretrained_ckpt[act_str])
                    if "model_state" in ckpt:
                        self.vo_model[act].load_state_dict(ckpt["model_state"])
                    else:
                        self.vo_model[act].load_state_dict(ckpt["model_states"][act])

        if self._run_type == "eval":
            logger.info(f"Eval {self.config.EVAL.EVAL_CKPT_PATH}")
            eval_ckpt = torch.load(self.config.EVAL.EVAL_CKPT_PATH)
            for act in self._act_list:
                self.vo_model[act].load_state_dict(eval_ckpt["model_states"][act])

        logger.info(self.vo_model[self._act_list[0]])
        num_param = sum(param.numel() for param in list(self.vo_model.values())[0].parameters() if param.requires_grad)
        logger.info("VO model's number of trainable parameters: {}".format(num_param))

        print("Detail model in training log")
        print("VO model's number of trainable parameters: {}".format(num_param))
        print("Starting ...")

    def _set_up_optimizer(self):
        self.optimizer = {}
        for act in self._act_list:
            self.optimizer[act] = optim.Adam(
                list(filter(lambda p: p.requires_grad, self.vo_model[act].parameters())),
                lr=self.config.VO.TRAIN.lr,
                eps=self.config.VO.TRAIN.eps,
                weight_decay=self.config.VO.TRAIN.weight_decay, )

    def _set_up_dataloader(self):
        self._data_collate_mode = "fast"
        collate_func = fast_collate_func
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        torch.cuda.manual_seed_all(self.config.TASK_CONFIG.SEED)
        self._train_batch_size = self.config.VO.TRAIN.batch_size

        if "discretized_depth" not in self._observation_space:
            assert self.config.VO.MODEL.discretize_depth == "none"
        if self.config.VO.MODEL.discretize_depth == "none":
            assert "discretized_depth" not in self._observation_space

        if "discretized_depth" in self._observation_space:
            self._discretized_depth_channels = (self.config.VO.MODEL.discretized_depth_channels)
        else:
            self._discretized_depth_channels = 0

        gen_top_down_view = False
        top_down_view_infos = {}
        if "top_down_view" in self._observation_space:
            gen_top_down_view = True
            top_down_view_infos = {
                "min_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
                "max_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                "vis_size_h": self.config.VO.VIS_SIZE_H,
                "vis_size_w": self.config.VO.VIS_SIZE_W,
                "hfov_rad": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV,
                "flag_center_crop": self.config.VO.MODEL.top_down_center_crop,
            }

        if self.train_loader is None:
            train_dataset = StatePairRegressionDataset(
                eval_flag=False,
                data_file=self.config.VO.DATASET.TRAIN,
                num_workers=TRAIN_NUM_WORKERS,
                act_type=self._act_type,
                vis_size_w=self.config.VO.VIS_SIZE_W,
                vis_size_h=self.config.VO.VIS_SIZE_H,
                collision=self.config.VO.TRAIN.collision,
                geo_invariance_types=self.geo_invariance_types,
                discretize_depth=self.config.VO.MODEL.discretize_depth,
                discretized_depth_channels=self._discretized_depth_channels,
                gen_top_down_view=gen_top_down_view,
                top_down_view_infos=top_down_view_infos,
                partial_data_n_splits=self.config.VO.DATASET.PARTIAL_DATA_N_SPLITS,
            )

            if isinstance(train_dataset, torch.utils.data.IterableDataset):
                shuffle_flag = False
            elif isinstance(train_dataset, torch.utils.data.Dataset):
                shuffle_flag = True

            self.train_loader = DataLoader(
                train_dataset,
                self._train_batch_size,
                shuffle=shuffle_flag,
                collate_fn=collate_func,
                num_workers=train_dataset.num_workers,
                drop_last=False,
                pin_memory=self._pin_memory_flag,
                timeout=TIMEOUT,
            )

        if self.eval_loader is None:
            eval_dataset = StatePairRegressionDataset(
                eval_flag=True,
                data_file=self.config.VO.DATASET.EVAL,
                num_workers=EVAL_NUM_WORKERS,
                act_type=train_dataset._act_type,
                vis_size_w=train_dataset._vis_size_w,
                vis_size_h=train_dataset._vis_size_h,
                collision=train_dataset._collision,
                geo_invariance_types=train_dataset._geo_invariance_types,
                discretize_depth=train_dataset._discretize_depth,
                discretized_depth_channels=train_dataset._discretized_depth_channels,
                gen_top_down_view=train_dataset._gen_top_down_view,
                top_down_view_infos=train_dataset._top_down_view_infos,
                partial_data_n_splits=1,
            )
            self.eval_loader = DataLoader(
                eval_dataset,
                EVAL_BATCHSIZE,
                collate_fn=collate_func,
                num_workers=eval_dataset.num_workers,
                drop_last=False,
                pin_memory=self._pin_memory_flag,
            )

    def _transfer_batch(self, act_idxes, rgb_pairs, depth_pairs, discretized_depth_pairs, top_down_view_pairs, ):

        batch_pairs = {}
        pair_list = [rgb_pairs, depth_pairs, discretized_depth_pairs, top_down_view_pairs]
        ob_list = ["rgb", "depth", "discretized_depth", "top_down_view"]
        if self._data_collate_mode == "fast":
            for i in range(len(ob_list)):
                ob = ob_list[i]
                if ob in self._observation_space:
                    batch_pairs[ob] = torch.cat(
                        [_.float().to(self.device, non_blocking=self._pin_memory_flag) for _ in pair_list[i]],
                        dim=0)[act_idxes, :]
        elif self._data_collate_mode == "normal":
            for i in range(len(ob_list)):
                ob = ob_list[i]
                if ob in self._observation_space:
                    batch_pairs[ob] = (
                        pair_list[i][act_idxes, :].float().to(self.device, non_blocking=self._pin_memory_flag))
        return batch_pairs

    def _compute_model_output(self, a, batch_pairs, act=-1):
        # [batch, 3], [delta_x, delta_z, delta_yaw]
        if "act_embed" in self.config.VO.MODEL.name:
            a = torch.squeeze(a, dim=1).to(self.device, non_blocking=self._pin_memory_flag)
            pred_delta_states = self.vo_model[act](batch_pairs, a)
        else:
            pred_delta_states = self.vo_model[act](batch_pairs)
        return pred_delta_states

    def _compute_geo_invariance_inverse_loss(self, deltas, a, dt):
        actions_deduplicate = a[torch.nonzero(dt == CUR_REL_TO_PREV, as_tuple=True)[0]]
        deltas_cur_rel_to_prev = deltas[torch.nonzero(dt == CUR_REL_TO_PREV, as_tuple=True)[0], :]
        deltas_prev_rel_to_cur = deltas[torch.nonzero(dt == PREV_REL_TO_CUR, as_tuple=True)[0], :]
        dyaw_prev_rel_to_cur = deltas_prev_rel_to_cur[:, 2]
        geo_inverse_rot_diffs = (deltas_cur_rel_to_prev[:, 2] + deltas_prev_rel_to_cur[:, 2]) ** 2
        loss_geo_inverse_rot = torch.mean(geo_inverse_rot_diffs)
        abs_diff_gir = torch.mean(torch.sqrt(geo_inverse_rot_diffs.detach()))
        rot_mat_prev_rel_to_cur = torch.stack(
            (
                torch.cos(dyaw_prev_rel_to_cur),
                torch.sin(dyaw_prev_rel_to_cur),
                -1 * torch.sin(dyaw_prev_rel_to_cur),
                torch.cos(dyaw_prev_rel_to_cur),
            ),
            dim=1,
        ).reshape((-1, 2, 2))
        pred_pos_prev_rel_to_cur = torch.matmul(rot_mat_prev_rel_to_cur,
                                                deltas_cur_rel_to_prev[:, :2].unsqueeze(-1), ).squeeze(-1)
        geo_inverse_pos_diffs = (deltas_prev_rel_to_cur[:, :2] + pred_pos_prev_rel_to_cur) ** 2
        forward_idxes = torch.nonzero(actions_deduplicate == MOVE_FORWARD, as_tuple=True)[0]
        if forward_idxes.size(0) != 0:
            mask = torch.ones(geo_inverse_pos_diffs.size()).to(geo_inverse_pos_diffs.device)
            mask[forward_idxes, 1] = 0.0
            geo_inverse_pos_diffs = mask * geo_inverse_pos_diffs
        loss_geo_inverse_pos = torch.mean(geo_inverse_pos_diffs)
        abs_diff_gip = torch.mean(torch.sqrt(geo_inverse_pos_diffs.detach()), dim=0)
        loss_geo_inverse = loss_geo_inverse_rot + loss_geo_inverse_pos
        return loss_geo_inverse, abs_diff_gir, abs_diff_gip

    def _process_one_batch(self, batch_data, cur_geo_invariance_types, abs_diffs, tm, relative_diffs, train_flag=True,
                           save_pred=False, gt_deltas_to_save={}, pred_deltas_to_save={}):
        (dt, r_rgb, r_d, r_dd, r_tdv, a, dxs, dys, dzs, d_yaws, d_m, c_idxs, e_idxs,) = batch_data

        if self._data_collate_mode == "fast":
            # explicitly deepcopy and reduce reference count to let dataloader processes close
            rgb_pairs = [_.clone() for _ in r_rgb]
            depth_pairs = [_.clone() for _ in r_d]
            discretized_depth_pairs = [_.clone() for _ in r_dd]
            top_down_view_pairs = [_.clone() for _ in r_tdv]
            del r_rgb[:]
            del r_d[:]
            del r_dd[:]
            del r_tdv[:]
            del r_rgb
            del r_d
            del r_dd
            del r_tdv
        elif self._data_collate_mode == "normal":
            rgb_pairs = r_rgb
            depth_pairs = r_d
            discretized_depth_pairs = r_dd
            top_down_view_pairs = r_tdv
        else:
            raise ValueError

        d_abs_diff_geo_ir = None
        d_abs_diff_geo_ip = None
        if "inverse_joint_train" in cur_geo_invariance_types and train_flag:
            debug_idxes = torch.nonzero((a == TURN_LEFT) | (a == TURN_RIGHT), as_tuple=True, )[0]
            all_gt_deltas = torch.cat((dxs, dzs, d_yaws), dim=1)
            (_, d_abs_diff_geo_ir, d_abs_diff_geo_ip,) = self._compute_geo_invariance_inverse_loss(
                all_gt_deltas[debug_idxes, :], a[debug_idxes, :], dt[debug_idxes, :], )

        if isinstance(self._act_type, int) and self._act_type != -1:
            assert torch.sum(a - self._act_type) == 0
        if isinstance(self._act_type, list):
            assert torch.sum(a == TURN_RIGHT) + torch.sum(a == TURN_LEFT) == a.size(0)
        a = a.to(self.device)
        dxs = dxs.to(self.device)
        dzs = dzs.to(self.device)
        d_yaws = d_yaws.to(self.device)
        d_m = d_m.to(self.device)
        cur_batch_size = dxs.size(0)
        loss_weights = self._compute_loss_weights(a, dxs, dzs, d_yaws)
        loss = 0.0
        if "inverse_joint_train" in cur_geo_invariance_types:
            idx_mfr = []
            apd = []
            aa = []
            adt = []
        for act in self._act_list:
            if act == -1:
                act_idxes = torch.arange(a.size(0))
            else:
                act_idxes = torch.nonzero(a == act, as_tuple=True)[0]
            loss_weights_cur_act = {k: v[act_idxes] for k, v in loss_weights.items()}

            batch_pairs = self._transfer_batch(act_idxes, rgb_pairs, depth_pairs, discretized_depth_pairs,
                                               top_down_view_pairs, )
            pred_delta_states = self._compute_model_output(a[act_idxes, :], batch_pairs, act=act, )

            if "inverse_joint_train" in cur_geo_invariance_types:
                idx_mfr.extend(
                    [
                        (i, j)
                        for i, j in zip(
                        act_idxes.cpu().numpy(),
                        len(idx_mfr) + np.arange(act_idxes.size(0)),
                    )
                    ]
                )
                apd.append(pred_delta_states)
                adt.append(dt[act_idxes])
                aa.append(a[act_idxes])

            if train_flag:
                if len(cur_geo_invariance_types) == 0:
                    abs_diffs[act] = []
                    tm[act] = []
                    relative_diffs[act] = []
                else:
                    abs_diffs[act] = defaultdict(list)
                    tm[act] = defaultdict(list)
                    relative_diffs[act] = defaultdict(list)

            for i, d_type in enumerate(self.delta_types):
                if len(cur_geo_invariance_types) == 0:
                    tmp_data_types = []
                    if save_pred and i == 0:
                        if act not in gt_deltas_to_save:
                            gt_deltas_to_save[act] = []
                            pred_deltas_to_save[act] = []
                        gt_deltas_to_save[act].append(
                            (
                                torch.cat(
                                    (c_idxs[act_idxes], e_idxs[act_idxes], dxs[act_idxes, :].cpu(),
                                     dzs[act_idxes, :].cpu(), d_yaws[act_idxes, :].cpu(),),
                                    dim=1,
                                ).cpu().numpy()
                            )
                        )
                        pred_deltas_to_save[act].append(
                            torch.cat(
                                (c_idxs[act_idxes], e_idxs[act_idxes], pred_delta_states[act_idxes, :].detach().cpu(),),
                                dim=1, ).numpy())

                    d_l = self._compute_and_update_info(
                        act,
                        d_type,
                        i,
                        pred_delta_states,
                        [
                            dxs[act_idxes],
                            dzs[act_idxes],
                            d_yaws[act_idxes],
                        ],
                        loss_weights_cur_act,
                        d_m[act_idxes],
                        abs_diffs,
                        tm,
                        relative_diffs,
                        update="append" if train_flag else "sum",
                        sum_multiplier=1 if train_flag else cur_batch_size,
                    )

                    loss += d_l
                else:
                    tmp_data_types = [CUR_REL_TO_PREV]
                    if "inverse_data_augment_only" in cur_geo_invariance_types or "inverse_joint_train" in cur_geo_invariance_types:
                        tmp_data_types.append(PREV_REL_TO_CUR)
                    for tmp_id in tmp_data_types:
                        tmp_name = DATA_TYPE_ID2STR[tmp_id]
                        tmp_idxes = torch.nonzero(dt[act_idxes] == tmp_id, as_tuple=True, )[0]
                        tmp_loss_weights = {k: v[tmp_idxes] for k, v in loss_weights_cur_act.items()}
                        if save_pred and i == 0:
                            if act not in gt_deltas_to_save:
                                gt_deltas_to_save[act] = {}
                                pred_deltas_to_save[act] = {}
                            if tmp_id not in gt_deltas_to_save[act]:
                                gt_deltas_to_save[act][tmp_id] = []
                                pred_deltas_to_save[act][tmp_id] = []

                            gt_deltas_to_save[act][tmp_id].append(
                                torch.cat(
                                    (
                                        c_idxs[act_idxes][tmp_idxes],
                                        e_idxs[act_idxes][tmp_idxes],
                                        dxs[act_idxes][tmp_idxes].cpu(),
                                        dzs[act_idxes][tmp_idxes].cpu(),
                                        d_yaws[act_idxes][tmp_idxes].cpu(),
                                    ),
                                    dim=1,
                                ).numpy()
                            )

                            pred_deltas_to_save[act][tmp_id].append(
                                torch.cat(
                                    (
                                        c_idxs[act_idxes][tmp_idxes],
                                        e_idxs[act_idxes][tmp_idxes],
                                        pred_delta_states[tmp_idxes, :].detach().cpu(),
                                    ),
                                    dim=1,
                                ).numpy()
                            )

                        d_l = self._compute_and_update_info(
                            act,
                            d_type,
                            i,
                            pred_delta_states[tmp_idxes, :],
                            [
                                dxs[act_idxes][tmp_idxes],
                                dzs[act_idxes][tmp_idxes],
                                d_yaws[act_idxes][tmp_idxes],
                            ],
                            tmp_loss_weights,
                            d_m[act_idxes][tmp_idxes],
                            abs_diffs,
                            tm,
                            relative_diffs,
                            update="append" if train_flag else "sum",
                            sum_multiplier=1 if train_flag else cur_batch_size,
                            data_type_name=tmp_name,
                        )

                        loss += d_l

        if "inverse_joint_train" in cur_geo_invariance_types:
            apd = torch.cat(apd, dim=0)
            aa = torch.cat(aa, dim=0)
            adt = torch.cat(adt, dim=0)
            idx_mfr = {_[0]: _[1] for _ in idx_mfr}
            aa = torch.stack([aa[idx_mfr[i], :] for i in np.arange(aa.size(0))], dim=0, )
            apd = torch.stack([apd[idx_mfr[i], :] for i in np.arange(apd.size(0))], dim=0, )
            adt = torch.stack([adt[idx_mfr[i], :] for i in np.arange(adt.size(0))], dim=0, )
            tmp = torch.nonzero((aa == TURN_LEFT) | (aa == TURN_RIGHT), as_tuple=True, )[0]
            (loss_geo_inverse, abs_diff_gir, abs_diff_gip,) = self._compute_geo_invariance_inverse_loss(apd[tmp, :],
                                                                                                        aa[tmp, :],
                                                                                                        adt[tmp, :], )
            loss += self.config.VO.GEOMETRY.loss_inv_weight * loss_geo_inverse
        else:
            abs_diff_gir = None
            abs_diff_gip = None

        infos_for_log = (
            abs_diffs,
            tm,
            relative_diffs,
            abs_diff_gir,
            abs_diff_gip,
            d_abs_diff_geo_ir,
            d_abs_diff_geo_ip,
        )

        return loss, cur_batch_size, batch_pairs, tmp_data_types, infos_for_log

    def train(self):
        start_epoch = 0
        nbatches = np.ceil(len(self.train_loader.dataset) / self._train_batch_size)
        grad_info_dict = OrderedDict()
        with (
                TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs)
        ) as writer:
            for epoch in tqdm(range(start_epoch, self.config.VO.TRAIN.epochs)):
                train_iter = iter(self.train_loader)
                batch_i = 0
                with tqdm(total=nbatches) as pbar:
                    while True:
                        try:
                            batch_data = next(train_iter)
                        except StopIteration:
                            break
                        batch_i += 1
                        pbar.update()
                        if batch_i >= nbatches:
                            nbatches += 1
                            pbar.total = nbatches
                            pbar.refresh()
                        global_step = batch_i + epoch * nbatches
                        for act in self._act_list:
                            self.optimizer[act].zero_grad()
                        with (autograd.detect_anomaly() if self.config.VO.debug == 1 else contextlib.suppress()):
                            abs_diffs = {}
                            tm = {}
                            relative_diffs = {}
                            loss, cur_batch_size, batch_pairs, tmp_data_types, infos_for_log, = self._process_one_batch(
                                batch_data,
                                self.geo_invariance_types,
                                abs_diffs,
                                tm,
                                relative_diffs,
                                train_flag=True,
                            )
                            (
                                abs_diffs,
                                tm,
                                relative_diffs,
                                abs_diff_gir,
                                abs_diff_gip,
                                d_abs_diff_geo_ir,
                                d_abs_diff_geo_ip,
                            ) = infos_for_log

                            loss.backward()

                            if self.config.VO.TRAIN.log_grad:
                                self._log_grad(writer, global_step, grad_info_dict, d_type=d_type)

                            for act in self._act_list:
                                self.optimizer[act].step()

                            self._log_lr(writer, global_step)

                            if batch_i == 10:
                                self._obs_log_func(writer, global_step, batch_pairs)

                            writer.add_scalar(f"Objective/train", loss, global_step=global_step)

                            if batch_i == nbatches - 1:
                                self._save_dict(
                                    {"train_objevtive": [loss.cpu().item()]},
                                    os.path.join(self.config.INFO_DIR, f"train_objective_info.p"),
                                )

                            for act in self._act_list:
                                for i, d_type in enumerate(self.delta_types):
                                    log_name = f"train_{ACT_IDX2NAME[act]}"
                                    if len(self.geo_invariance_types) == 0:
                                        self._regress_log_func(
                                            writer,
                                            log_name,
                                            global_step,
                                            abs_diffs[act][i],
                                            tm[act][i],
                                            relative_diffs[act][i],
                                            d_type=d_type,
                                        )

                                        if batch_i == nbatches - 1:
                                            self._regress_udpate_dict(
                                                log_name,
                                                abs_diffs[act][i],
                                                tm[act][i],
                                                relative_diffs[act][i],
                                                d_type=d_type,
                                            )
                                    else:
                                        for tmp_id in tmp_data_types:
                                            tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                            self._regress_log_func(
                                                writer,
                                                f"{log_name}_{tmp_name}",
                                                global_step,
                                                abs_diffs[act][tmp_name][i],
                                                tm[act][tmp_name][i],
                                                relative_diffs[act][tmp_name][i],
                                                d_type=d_type,
                                            )

                                            if batch_i == nbatches - 1:
                                                self._regress_udpate_dict(
                                                    f"{log_name}_{tmp_name}",
                                                    abs_diffs[act][tmp_name][i],
                                                    tm[act][tmp_name][i],
                                                    relative_diffs[act][tmp_name][i],
                                                    d_type=d_type,
                                                )
                            if "inverse_joint_train" in self.geo_invariance_types:
                                self._geo_invariance_inverse_log_func(
                                    writer,
                                    "train",
                                    global_step,
                                    abs_diff_gir,
                                    abs_diff_gip,
                                )

                                self._geo_invariance_inverse_log_func(
                                    writer,
                                    "train_debug",
                                    global_step,
                                    d_abs_diff_geo_ir,
                                    d_abs_diff_geo_ip,
                                )

                                if batch_i == nbatches - 1:
                                    self._geo_invariance_inverse_udpate_dict(
                                        "train",
                                        abs_diff_gir,
                                        abs_diff_gip,
                                    )

                time.sleep(2)

                self.eval(eval_act="no_specify", epoch=epoch + 1, writer=writer, split_name="eval_all", )
                for act in self._act_list:
                    self.vo_model[act].train()
                self._save_ckpt(epoch + 1)

    def eval(self, eval_act="no_specify", epoch=0, writer=None, split_name="eval", save_pred=False, **kwargs):

        for act in self._act_list:
            self.vo_model[act].eval()

        if eval_act == "no_specify":
            eval_loader = self.eval_loader
        else:
            if eval_act in self.separate_eval_loaders:
                eval_loader = self.separate_eval_loaders[eval_act]
            else:
                raise ValueError

        eval_geo_invariance_types = eval_loader.dataset.geo_invariance_types

        total_size = 0
        total_loss = 0.0

        total_abs_diffs = {}
        total_target_magnitudes = {}
        total_relative_diffs = {}
        for act in self._act_list:
            if len(eval_geo_invariance_types) == 0:
                total_abs_diffs[act] = defaultdict(float)
                total_target_magnitudes[act] = defaultdict(float)
                total_relative_diffs[act] = defaultdict(float)
            else:
                total_abs_diffs[act] = defaultdict(lambda: defaultdict(float))
                total_target_magnitudes[act] = defaultdict(lambda: defaultdict(float))
                total_relative_diffs[act] = defaultdict(lambda: defaultdict(float))

        t_abs_diff_geo_gir = 0.0
        t_abs_diff_geo_gip = 0.0

        gt_deltas = {}
        pred_deltas = {}

        nbatches = np.ceil(len(eval_loader.dataset) / EVAL_BATCHSIZE)
        with tqdm(total=nbatches) as pbar:
            with torch.no_grad():
                eval_iter = iter(eval_loader)
                batch_i = 0
                while True:
                    try:
                        batch_data = next(eval_iter)
                    except StopIteration:
                        break
                    batch_i += 1
                    pbar.update()

                    if batch_i > nbatches:
                        nbatches += 1
                        pbar.total = nbatches
                        pbar.refresh()

                    (
                        loss,
                        cur_batch_size,
                        batch_pairs,
                        tmp_data_types,
                        infos_for_log,
                    ) = self._process_one_batch(
                        batch_data,
                        eval_geo_invariance_types,
                        total_abs_diffs,
                        total_target_magnitudes,
                        total_relative_diffs,
                        train_flag=False,
                        save_pred=save_pred,
                        gt_deltas_to_save=gt_deltas,
                        pred_deltas_to_save=pred_deltas,
                    )
                    total_loss += loss * cur_batch_size
                    total_size += cur_batch_size

                    (
                        abs_diffs,
                        tm,
                        relative_diffs,
                        abs_diff_gir,
                        abs_diff_gip,
                        d_abs_diff_geo_ir,
                        d_abs_diff_geo_ip,
                    ) = infos_for_log

                    if "inverse_joint_train" in eval_geo_invariance_types:
                        t_abs_diff_geo_gir += (abs_diff_gir * cur_batch_size / 2)
                        t_abs_diff_geo_gip += (abs_diff_gip * cur_batch_size / 2)

                target_size = len(eval_loader.dataset)
                if "inverse_joint_train" in eval_geo_invariance_types:
                    target_size += len(eval_loader.dataset)
                del eval_iter
                time.sleep(2)
                if save_pred:
                    for act in gt_deltas:
                        if "inverse_joint_train" in eval_geo_invariance_types or "inverse_data_augment_only" in eval_geo_invariance_types:
                            for tmp_id in gt_deltas[act]:
                                pred_deltas[act][tmp_id] = np.concatenate(pred_deltas[act][tmp_id], axis=0)
                                gt_deltas[act][tmp_id] = np.concatenate(gt_deltas[act][tmp_id], axis=0)
                        else:
                            pred_deltas[act] = np.concatenate(pred_deltas[act], axis=0)
                            gt_deltas[act] = np.concatenate(gt_deltas[act], axis=0)

                    with open(os.path.join(self.config.LOG_DIR, f"delta_gt_pred.p"), "wb") as f:
                        joblib.dump({"gt": dict(gt_deltas), "pred": dict(pred_deltas)}, f, compress="lz4")

                if writer is not None:
                    writer.add_scalar(f"Objective/{split_name}", total_loss / total_size, global_step=epoch)

                    for act in self._act_list:
                        for d_type in self.delta_types:
                            log_name = f"{split_name}_{ACT_IDX2NAME[act]}"
                            if len(eval_geo_invariance_types) == 0:
                                self._regress_log_func(
                                    writer,
                                    log_name,
                                    epoch,
                                    total_abs_diffs[act][d_type] / total_size,
                                    total_target_magnitudes[act][d_type] / total_size,
                                    total_relative_diffs[act][d_type] / total_size,
                                    d_type=d_type,
                                )
                            else:
                                for tmp_id in tmp_data_types:
                                    tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                    self._regress_log_func(
                                        writer,
                                        f"{log_name}_{tmp_name}",
                                        epoch,
                                        total_abs_diffs[act][tmp_name][d_type] / total_size,
                                        total_target_magnitudes[act][tmp_name][d_type] / total_size,
                                        total_relative_diffs[act][tmp_name][d_type] / total_size,
                                        d_type=d_type,
                                    )
                    if "inverse_joint_train" in eval_geo_invariance_types:
                        self._geo_invariance_inverse_log_func(
                            writer,
                            split_name,
                            epoch,
                            t_abs_diff_geo_gir / (total_size / 2),
                            t_abs_diff_geo_gip / (total_size / 2),
                        )
                self._save_dict(
                    {f"{split_name}_objective": [(epoch, total_loss.cpu().item() / total_size)]},
                    os.path.join(self.config.INFO_DIR, f"eval_objective_info.p"),
                )

                for act in self._act_list:
                    for d_type in self.delta_types:
                        log_name = f"{split_name}_{ACT_IDX2NAME[act]}"

                        if len(eval_geo_invariance_types) == 0:
                            self._regress_udpate_dict(
                                log_name,
                                total_abs_diffs[act][d_type] / total_size,
                                total_target_magnitudes[act][d_type] / total_size,
                                total_relative_diffs[act][d_type] / total_size,
                                d_type=d_type,
                            )
                        else:
                            for tmp_id in tmp_data_types:
                                tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                self._regress_udpate_dict(
                                    f"{log_name}_{tmp_name}",
                                    total_abs_diffs[act][tmp_name][d_type] / total_size,
                                    total_target_magnitudes[act][tmp_name][d_type] / total_size,
                                    total_relative_diffs[act][tmp_name][d_type] / total_size,
                                    d_type=d_type,
                                )

                if "inverse_joint_train" in eval_geo_invariance_types:
                    self._geo_invariance_inverse_udpate_dict(
                        split_name,
                        t_abs_diff_geo_gir / (total_size / 2),
                        t_abs_diff_geo_gip / (total_size / 2),
                    )

    def _compute_and_update_info(
            self,
            act,
            d_type,
            delta_idx,
            pred_delta_states,
            gt_deltas,
            loss_weights,
            d_m,
            abs_diffs,
            tm,
            relative_diffs,
            update="append",
            sum_multiplier=1,
            data_type_name=None,
    ):
        d_l, d_a_d, d_tm, d_rd = self._compute_loss(pred_delta_states[:, delta_idx].unsqueeze(1), gt_deltas,
                                                    d_type=d_type, loss_weights=loss_weights, d_m=d_m)
        if update == "sum":
            if data_type_name is None:
                abs_diffs[act][d_type] += d_a_d * sum_multiplier
                tm[act][d_type] += d_tm * sum_multiplier
                relative_diffs[act][d_type] += d_rd * sum_multiplier
            else:
                abs_diffs[act][data_type_name][d_type] += d_a_d * sum_multiplier
                tm[act][data_type_name][d_type] += (d_tm * sum_multiplier)
                relative_diffs[act][data_type_name][d_type] += (d_rd * sum_multiplier)
        elif update == "append":
            if data_type_name is None:
                abs_diffs[act].append(d_a_d)
                tm[act].append(d_tm)
                relative_diffs[act].append(d_rd)
            else:
                abs_diffs[act][data_type_name].append(d_a_d)
                tm[act][data_type_name].append(d_tm)
                relative_diffs[act][data_type_name].append(d_rd)
        return d_l

    def _log_lr(self, writer, global_step):
        for tmp_i, param_group in enumerate(self.optimizer[self._act_list[0]].param_groups):
            writer.add_scalar(f"LR/group_{tmp_i}", param_group["lr"], global_step=global_step, )

    def _geo_invariance_inverse_log_func(self, writer, split, global_step, abs_diff_gir, abs_diff_gip, ):
        writer.add_scalar(f"{split}_geo_invariance/inverse_abs_diff_dyaw", abs_diff_gir, global_step=global_step)
        writer.add_scalar(f"{split}_geo_invariance/inverse_abs_diff_dx", abs_diff_gip[0], global_step=global_step)
        writer.add_scalar(f"{split}_geo_invariance/inverse_abs_diff_dz", abs_diff_gip[1], global_step=global_step)

    def _geo_invariance_inverse_udpate_dict(self, split, abs_diff_gir, abs_diff_gip):

        info_dict = defaultdict(list)
        info_dict[f"geo_invariance_inverse_abs_diff_dz"].append(abs_diff_gip[1].item())
        info_dict[f"geo_invariance_inverse_abs_diff_dx"].append(abs_diff_gip[0].item())
        info_dict[f"geo_invariance_inverse_abs_diff_dyaw"].append(abs_diff_gir.item())
        save_f = os.path.join(self.config.INFO_DIR, f"{split}_invariance_info.p")
        self._save_dict(dict(info_dict), save_f)

    def _obs_log_func(self, writer, global_step, batch_pairs, ):
        if "rgb" in self._observation_space:
            writer.add_image("pre_obs/rgb", batch_pairs["rgb"][0, :, :, :3].cpu(), global_step, dataformats="HWC", )
        if "depth" in self._observation_space:
            writer.add_image("pre_obs/depth", batch_pairs["depth"][0, :, :, 0].cpu(), global_step, dataformats="HW", )
            writer.add_image("cur_obs/depth", batch_pairs["depth"][0, :, :, 1].cpu(), global_step, dataformats="HW", )
        if "discretized_depth" in self._observation_space:
            for i in range(self.config.VO.MODEL.discretized_depth_channels):
                tmp_name = f"pre_obs/discretized_depth_{i}"
                writer.add_image(tmp_name, batch_pairs["discretized_depth"][0, :, :, i].cpu(), global_step,
                                 dataformats="HW")
                writer.add_image(
                    tmp_name,
                    batch_pairs["discretized_depth"][0, :, :,
                    self.config.VO.MODEL.discretized_depth_channels + i].cpu(),
                    global_step,
                    dataformats="HW", )
        if "top_down_view" in self._observation_space:
            writer.add_image("prev_obs/tdv", batch_pairs["top_down_view"][0, :, :, 0].cpu(), global_step,
                             dataformats="HW")
            writer.add_image("cur_obs/tdv", batch_pairs["top_down_view"][0, :, :, 1].cpu(), global_step,
                             dataformats="HW")

    def _save_ckpt(self, epoch):
        state = {
            "epoch": epoch,
            "config": self.config,
            "model_states": {k: self.vo_model[k].state_dict() for k in self._act_list},
            "optim_states": {k: self.optimizer[k].state_dict() for k in self._act_list},
            "rnd_state": random.getstate(),
            "np_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
            "torch_cuda_rnd_state": torch.cuda.get_rng_state_all(),
        }
        try:
            torch.save(state, os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt_epoch_{epoch}.pth"), )
        except:
            print(f"\ntotal size: {sys.getsizeof(state)}")
            for k, v in state.items():
                print(f"size of {k}: {sys.getsizeof(v)}")
