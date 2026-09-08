import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
from torch.optim.optimizer import required

from only_train_once.transform import (
    TensorTransform,
    tensor_transformation_param_group,
)

from .base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class GETA(BaseHybridSparseOptimizer):
    """
    GETA: General and Efficient Training framework that Automates
    joint structured pruning and quantization.
    """

    def __init__(
        self,
        params,
        variant="sgd",
        lr=required,
        lr_quant=1e-3,
        first_momentum=None,
        second_momentum=None,
        dampening=None,
        weight_decay=None,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_steps=1,
        projection_periods=1,
        start_pruning_step=1,
        pruning_steps=1,
        pruning_periods=1,
        group_divisible=1,
        importance_score_criteria="default",
        bit_reduction=2,
        min_bit_wt=2,
        max_bit_wt=16,
        min_bit_act=2,
        max_bit_act=16,
        grad_clip_min=-1.0,
        grad_clip_max=1.0,
        verbose="False",  # if verbose="False", no information is printed
        device="cuda",
        log_dir="outputs",
    ):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start_projection_step = start_projection_step
        self.projection_steps = projection_steps
        self.projection_periods = projection_periods
        self.projection_period_duration = (
            self.projection_steps // self.projection_periods
        )
        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(
            max(1, pruning_periods)
        )  # How many periods that the pruning last for.
        self.pruning_steps = pruning_steps
        self.pruning_period_duration = (
            self.pruning_steps // self.pruning_periods
        )  # How many pruning steps for each period
        self.curr_pruning_period = 0  # Track pruning period
        self.lr_quant = lr_quant
        self.bit_reduction = bit_reduction
        self.min_bit_wt = min_bit_wt  # Minimum bit width for weights
        self.max_bit_wt = max_bit_wt  # Maximum bit width for weights
        self.min_bit_act = min_bit_act  # Minimum bit width for activations
        self.max_bit_act = max_bit_act  # Maximum bit width for activations
        self.grad_clip_min = grad_clip_min
        self.grad_clip_max = grad_clip_max
        self.verbose = verbose
        self.device = device
        self.pruned_group_idxes = list()
        self.gamma = 0.0
        self.d_quant = 0.0
        self.bit_layers = {}  # Store the bit width for each layer

        if importance_score_criteria == "default":
            self.importance_score_criteria = {
                "magnitude": 0.2,
                "avg_magnitude": 0.2,
                "cosine_similarity": 0.2,
                "taylor_first_order": 0.2,
                "taylor_second_order": 0.2,
            }
        else:
            self.importance_score_criteria = importance_score_criteria
        self.logger.info("Setup GETA")
        self.logger.info(f"importance_score_criteria: {self.importance_score_criteria}")
        self.logger.info(f"start_projection_step: {start_projection_step}")
        self.logger.info(f"projection_steps: {projection_steps}")
        self.logger.info(f"projection_periods: {projection_periods}")
        self.logger.info(f"start_pruning_step: {start_pruning_step}")
        self.logger.info(f"pruning_steps: {pruning_steps}")
        self.logger.info(f"pruning_periods: {self.pruning_periods}")
        self.logger.info(f"pruning_period_duration: {self.pruning_period_duration}")

        super().__init__(
            params=params,
            variant=variant,
            lr=lr,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            target_group_sparsity=target_group_sparsity,
            group_divisible=group_divisible,
        )

        for param_group in self.param_groups:
            param_group["important_idxes"] = [
                i for i in range(param_group["num_groups"])
            ]
            param_group["active_redundant_idxes"] = list()
            param_group["pruned_idxes"] = list()
            param_group["importance_scores"] = dict()
            param_group["lr_quant"] = lr_quant

        self.active_num_redundant_groups = list()
        # Set up active number redundant groups for each pruning period
        groups_sum = 0
        for p in range(self.pruning_periods):
            if p == self.pruning_periods - 1:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                groups_sum += self.active_num_redundant_groups[p]
        self.logger.info(
            f"Target redundant groups per period: {self.active_num_redundant_groups}"
        )

    @contextmanager
    def safe_open_file(self, filename, mode="a"):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            file = open(filename, mode)
            yield file
        except OSError as e:
            self.logger.error(f"Error opening file {filename}: {e}")
        finally:
            file.close()

    def grad_clipping(self):
        grad_clip_min = self.grad_clip_min
        grad_clip_max = self.grad_clip_max
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = p.grad.clamp(min=grad_clip_min, max=grad_clip_max)

    def identify_redundant_groups(self):
        global_scores = torch.cat(self.global_scores, dim=0)
        curr_active_num_redundant_groups = self.active_num_redundant_groups[
            self.curr_pruning_period
        ]
        curr_K = len(self.pruned_group_idxes) + curr_active_num_redundant_groups
        _, top_indices = torch.topk(-global_scores, curr_K)
        top_indices = top_indices.cpu().numpy()
        top_indices = np.setdiff1d(top_indices, self.pruned_group_idxes)[
            :curr_active_num_redundant_groups
        ].tolist()
        self.pruned_group_idxes.extend(top_indices)

        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                global_active_redundant_idx = np.intersect1d(
                    top_indices, group["global_idxes"]
                )
                group["active_redundant_idxes"] = (
                    global_active_redundant_idx - group["global_start_idx"]
                ).tolist()
                # Refine important_idx by group_divisible
                if group["num_groups"] < self.group_divisible:
                    group["active_redundant_idxes"].clear()
                    group["pruned_idxes"].clear()
                else:
                    curr_num_important_groups = len(group["important_idxes"])
                    trial_num_important_groups = curr_num_important_groups - len(
                        group["active_redundant_idxes"]
                    )
                    if (
                        trial_num_important_groups % self.group_divisible != 0
                        or trial_num_important_groups <= 0
                    ):
                        ratio = (
                            trial_num_important_groups // self.group_divisible + 1
                        )  # Add one will preserve more groups, otherwise will slim more.
                        refined_num_important_groups = None
                        if ratio <= 1 or trial_num_important_groups == 0:
                            refined_num_important_groups = max(
                                int(self.group_divisible), 1
                            )
                        else:
                            refined_num_important_groups = max(
                                int(ratio * self.group_divisible),
                                int(self.group_divisible),
                            )
                        refined_num_important_groups = min(
                            group["num_groups"], refined_num_important_groups
                        )
                        refined_num_active_redundant_groups = (
                            group["num_groups"]
                            - len(group["pruned_idxes"])
                            - refined_num_important_groups
                        )
                        self.target_num_redundant_groups += (
                            refined_num_active_redundant_groups
                            - len(group["active_redundant_idxes"])
                        )
                        group["active_redundant_idxes"] = group[
                            "active_redundant_idxes"
                        ][:refined_num_active_redundant_groups]
                group["important_idxes"] = [
                    i
                    for i in group["important_idxes"]
                    if (
                        i not in group["active_redundant_idxes"]
                        and i not in group["pruned_idxes"]
                    )
                ]

    def commit_redundant_idxes(self):
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                group["pruned_idxes"].extend(group["active_redundant_idxes"].copy())
                group["active_redundant_idxes"].clear()
                group["important_idxes"] = [
                    i
                    for i in range(group["num_groups"])
                    if i not in group["pruned_idxes"]
                ]
                group["importance_scores"].clear()

    def quantize_weight(self, param_group, target_name):
        is_quantize = False  # Check if the "target_name" layer involves quantization
        t_quant = None
        for p_name in param_group["p_names"]:
            if "d_quant_wt" in p_name:
                layer_name = ".".join(p_name.split(".")[:-1])
                if layer_name in target_name and "weight" in target_name:
                    is_quantize = True
                    quantize_layer_name = layer_name
                    break

        if not is_quantize:
            return is_quantize, None
        else:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if quantize_layer_name in p_name and "d_quant_wt" in p_name:
                    d_quant = p.data
                if quantize_layer_name in p_name and "t_quant_wt" in p_name:
                    t_quant = p.data
                if quantize_layer_name in p_name and "q_m_wt" in p_name:
                    q_m = p.data
                if quantize_layer_name in p_name and "weight" in p_name:
                    weight = p.data

            with torch.no_grad():
                quantized_weight = self._quantize_helper(
                    weight, d_quant, q_m, t_quant=t_quant
                )

            return is_quantize, quantized_weight

    def compute_gamma_d(self, param_group, active_redundant_idxes, bit_range):
        t_quant = None
        qm_list = []
        layer_name_list = []  # Store layers with quantization mapping
        prune_param_clip_list = []  # Store prunable parameter clipping values
        prune_param_grad_list = []  # Store prunable parameter gradients
        prune_param_res_list = []  # Store prunable parameter residual values
        prune_param_clip_redundant_list = []
        prune_param_res_redundant_list = []
        prune_param_grad_redundant_list = []

        ###########################################
        ####  Layers with quantization mapping ####
        ###########################################
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "d_quant_wt" in p_name:
                        d_quant = p.data
                    if "t_quant_wt" in p_name:
                        t_quant = p.data
                    if "q_m_wt" in p_name:
                        q_m = p.data
                        qm_list.append(q_m.item())
                    if "weight" in p_name:
                        weight = p.data
            for p_name, p, p_transform in zip(
                param_group["p_names"],
                param_group["params"],
                param_group["p_transform"],
            ):
                if layer_name in p_name and "weight" in p_name:
                    clipped_weight = self._clip_helper(weight, q_m, t_quant=t_quant)
                    prune_param_clip_list.append(clipped_weight.data)
                    residual_weight = self._residual_helper(
                        weight, d_quant, q_m, t_quant=t_quant
                    )
                    prune_param_res_list.append(residual_weight)
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])
                elif (
                    layer_name in p_name and p_transform != 1
                ):  # bias in quantization mapping
                    prune_param_clip_list.append(p.data)
                    prune_param_res_list.append(
                        torch.tensor([0.0]).to(p.device).expand_as(p.data)
                    )
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])

        ###########################################
        ### Layers without quantization mapping ###
        ###########################################
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if not any(layer_name in p_name for layer_name in layer_name_list):
                prune_param_clip_list.append(p.data)
                prune_param_res_list.append(
                    torch.tensor([0.0]).to(p.device).expand_as(p.data)
                )
                prune_param_grad_list.append(param_group["grad_variant"][p_name])

        # Access values at redundant indices
        for i in range(len(prune_param_clip_list)):
            prune_param_clip_redundant_list.append(
                prune_param_clip_list[i][active_redundant_idxes]
            )
            prune_param_res_redundant_list.append(
                prune_param_res_list[i][active_redundant_idxes]
            )
            prune_param_grad_redundant_list.append(
                prune_param_grad_list[i][active_redundant_idxes]
            )

        # Get flattened value with its norm
        flatten_clip = torch.cat(
            [tensor.flatten() for tensor in prune_param_clip_redundant_list]
        )
        flatten_grad = torch.cat(
            [tensor.flatten() for tensor in prune_param_grad_redundant_list]
        )
        flatten_res = torch.cat(
            [tensor.flatten() for tensor in prune_param_res_redundant_list]
        )
        flatten_clip_norm = torch.norm(flatten_clip, p=2)
        flatten_grad_norm = torch.norm(flatten_grad, p=2)
        flatten_res_norm = torch.norm(flatten_res, p=2)

        eps = 1e-8
        cosine_similarity_clip = torch.div(
            torch.dot(flatten_clip, flatten_grad),
            torch.max(flatten_clip_norm, torch.tensor(eps).to(flatten_clip.device))
            * flatten_grad_norm,
        )
        cosine_similarity_res = torch.div(
            torch.dot(flatten_res, flatten_grad),
            torch.max(flatten_res_norm, torch.tensor(eps).to(flatten_res.device))
            * flatten_grad_norm,
        )

        eta = 0.999
        zeta = 0.9
        if torch.mean(flatten_clip).item() < 1e-8:
            forget_rate = 0.0
        else:
            if torch.isinf(cosine_similarity_clip) or torch.isnan(
                cosine_similarity_clip
            ):
                self.logger.warning(
                    "cosine_similarity_clip is inf or nan, setting forget rate to 0.0"
                )
                forget_rate = 0.0
            elif cosine_similarity_clip >= 0.0 and cosine_similarity_clip <= 1.0:
                t = (
                    self.num_steps - self.start_pruning_step
                ) % self.pruning_period_duration
                forget_rate = 1.0 - (self.pruning_period_duration - t - 1.0) / (
                    self.pruning_period_duration - t
                )
            elif cosine_similarity_clip >= -1.0 and cosine_similarity_clip < 0.0:
                forget_rate = (
                    -(1 - eta)
                    * param_group["lr"]
                    * flatten_grad_norm
                    / (cosine_similarity_clip * flatten_clip_norm)
                )
            else:
                # TODO: @xiaoyi, refactor
                if self.verbose == "True":
                    self.logger.warning(
                        f"Unexpected cosine_similarity_clip value: {cosine_similarity_clip}"
                    )
                    outID = "log_info_" + str(self.start_pruning_step)
                    filename = os.path.join(
                        "outputs",
                        f"sparsity_{self.target_group_sparsity * 100}",
                        f"{outID}.txt",
                    )
                    with self.safe_open_file(filename) as logfile:
                        logfile.write("Throw an error: cosine_similarity_clip error\n")
                        logfile.write(
                            f"similarity_value: {cosine_similarity_clip.item():^8.9e}\n"
                        )
                        logfile.write(
                            f"flatten_grad_max: {torch.max(flatten_grad).item():^8.9e}\n"
                        )
                        logfile.write(
                            f"flatten_clip_max: {torch.max(flatten_clip).item():^8.9e}\n"
                        )
                        logfile.write(
                            f"flatten_grad_mean: {torch.mean(flatten_grad).item():^8.9e}\n"
                        )
                        logfile.write(
                            f"flatten_clip_mean: {torch.mean(flatten_clip).item():^8.9e}\n"
                        )
                        # logfile.write("all_grad_max: {flatten_grad:^8.9e}\n".format(flatten_grad=torch.max(torch.Tensor(prune_param_grad_list))) )
                    self.logger.error("Error with computing cosine_similarity_clip!")
                    assert 1 == 2

        # Determine d_quant range
        bit_width_lower = bit_range[0]
        bit_width_upper = bit_range[1]
        d_quant_upper = self._d_quant_helper(
            bit_width_lower, max(np.abs(qm_list)), t_quant
        )
        d_quant_lower = self._d_quant_helper(
            bit_width_upper, max(np.abs(qm_list)), t_quant
        )

        # Safeguard mechanism for d_quant
        if cosine_similarity_res >= 0.0 or forget_rate == 0.0:
            d_quant = d_quant_upper
        else:
            d_quant = (
                -zeta
                * eta
                * param_group["lr"]
                * flatten_grad_norm
                / (forget_rate * cosine_similarity_res * flatten_res_norm)
            )
            while d_quant < d_quant_lower:  # Avoid quant step size d being too small.
                forget_rate = forget_rate * 0.8
                d_quant = d_quant / 0.8
            d_quant = min(
                d_quant_upper, d_quant
            )  # Avoid quant step size d being too large.
        if self.verbose == "True":
            filename = os.path.join(
                "outputs",
                f"sparsity_{self.target_group_sparsity * 100}",
                f"{outID}.txt",
            )
            with self.safe_open_file(filename) as logfile:
                content = "Step: {num_step:^11s} Layer_name: {name:^30s} clip_max: {clip_max:^8.5e} res_max: {res_max:^8.5e} grad_max: {grad_max:^8.5e} clip_grad: {clip_grad:^8.5e} res_grad: {res_grad:^8.5e} flatten_clip_norm: {flatten_clip_norm:^8.5e} flatten_res_norm: {flatten_res_norm:^8.5e} flatten_grad_norm: {flatten_grad_norm:^8.5e} cos(gamma): {angle_gamma:^8.5e} cos(d):{angle_d:^8.5e} forget_rate: {gamma:^8.5e} d_quant: {d_quant:^8.5e} \n".format(
                    num_step=str(self.num_steps),
                    name=layer_name_list[0] if layer_name_list else "N/A",
                    clip_max=torch.max(flatten_clip).item(),
                    res_max=torch.max(flatten_res).item(),
                    grad_max=torch.max(flatten_grad).item(),
                    clip_grad=torch.dot(flatten_clip, flatten_grad).item(),
                    res_grad=torch.dot(flatten_res, flatten_grad).item(),
                    flatten_clip_norm=flatten_clip_norm,
                    flatten_res_norm=flatten_res_norm,
                    flatten_grad_norm=flatten_grad_norm,
                    angle_gamma=cosine_similarity_clip,
                    angle_d=cosine_similarity_res,
                    gamma=forget_rate,
                    d_quant=d_quant,
                )
                logfile.write(content)

        return forget_rate, d_quant

    def get_bitwidth_dict(self, param_group):
        d_quant_wt = None
        q_m_wt = None
        t_quant_wt = None
        d_quant_act = None
        q_m_act = None
        t_quant_act = None
        layer_name_list = []
        bit_dict = {}
        for p_name in param_group["p_names"]:
            if "d_quant" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data
                    if "d_quant_wt" in p_name:
                        d_quant_wt = p.data
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
                    if "d_quant_act" in p_name:
                        d_quant_act = p.data

            bit_dict[layer_name] = {}
            bit_width_wt = self._bit_width_helper(
                d_quant=d_quant_wt, q_m=q_m_wt, t_quant=t_quant_wt
            )
            bit_width_act = self._bit_width_helper(
                d_quant=d_quant_act, q_m=q_m_act, t_quant=t_quant_act
            )

            if bit_width_wt is not None:
                bit_dict[layer_name]["weight"] = round(bit_width_wt)

            if bit_width_act is not None:
                bit_dict[layer_name]["activation"] = round(bit_width_act)

        return bit_dict

    def gradient_descent_step(self, param_group):
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

    def partial_projected_gradient_descent_step_range_wt(self, param_group):
        """Apply projected gradient descent for weight quantization parameters."""
        # First part remains the same - gradient descent step
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if (
                    "d_quant_wt" in p_name
                    or "t_quant_wt" in p_name
                    or "q_m_wt" in p_name
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            # Assign different learning rate to weights and quantization params
            if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        # Find layers in each param_group
        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # Projection
        for layer_name in layer_name_list:
            t_quant_wt = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data

            # Calculate bounds for this layer
            d_quant_min = self._d_quant_helper(self.max_bit_wt, q_m_wt, t_quant_wt)
            d_quant_max = self._d_quant_helper(self.min_bit_wt, q_m_wt, t_quant_wt)

            # Convert bounds to scalars if they're tensors
            if isinstance(d_quant_min, torch.Tensor):
                d_quant_min = float(d_quant_min.item())
            if isinstance(d_quant_max, torch.Tensor):
                d_quant_max = float(d_quant_max.item())

            # Apply bounds to d_quant parameters
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_wt" in p_name:
                    # Clip using scalar bounds
                    p.data.clamp_(min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_range_act(self, param_group):
        # Add them as hyperparameter in hesso_quant optimizer in the future
        min_bit = self.min_bit_act
        max_bit = self.max_bit_act

        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if (
                    "d_quant_act" in p_name
                    or "t_quant_act" in p_name
                    or "q_m_act" in p_name
                ):
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )

            # Assign learning rate to activation quantization params
            if (
                "d_quant_act" in p_name
                or "t_quant_act" in p_name
                or "q_m_act" in p_name
            ):
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )

        # Find layers in each param_group
        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_act" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        # Projection
        for layer_name in layer_name_list:
            t_quant_act = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_act" in p_name:
                    d_quant_min = self._d_quant_helper(max_bit, q_m_act, t_quant_act)
                    d_quant_max = self._d_quant_helper(min_bit, q_m_act, t_quant_act)
                    p.data = torch.clip(p.data, min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_fix(self, param_group, bit_dict):
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            # Assign different learning rate to weights and quantization params
            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        for layer_name in bit_dict.keys():
            t_quant_wt = None
            t_quant_act = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_wt" in p_name:
                    bit_width = bit_dict[layer_name]["weight"]
                    d_quant_wt = self._d_quant_helper(bit_width, q_m_wt, t_quant_wt)
                    p.data = torch.clip(p.data, min=d_quant_wt, max=d_quant_wt)
                if layer_name in p_name and "d_quant_act" in p_name:
                    bit_width = bit_dict[layer_name]["activation"]
                    d_quant_act = self._d_quant_helper(bit_width, q_m_act, t_quant_act)
                    p.data = torch.clip(p.data, min=d_quant_act, max=d_quant_act)

    @staticmethod
    def _bit_width_helper(d_quant=None, q_m=None, t_quant=None):
        if d_quant is None:
            return None

        if t_quant is None:
            t_quant = 1.0
        bit_width = (
            math.log2(math.exp(t_quant * math.log(abs(q_m))) / abs(d_quant) + 1) + 1
        )

        return bit_width

    @staticmethod
    def _d_quant_helper(bit_width, q_m, t_quant):
        """Calculate d_quant, using max absolute value of q_m for uniform quantization."""
        if t_quant is None:
            t_quant = 1.0

        # Get maximum absolute value if q_m is a tensor
        if isinstance(q_m, torch.Tensor):
            q_m = torch.max(torch.abs(q_m)).item()
        else:
            q_m = abs(q_m)
        # Prevent exact zero
        q_m = max(abs(q_m), 1e-10)

        # Calculate d_quant using scalar math
        # d_quant = math.exp(t_quant * math.log(q_m)) / (2 ** (bit_width - 1) - 1)
        d_quant = math.exp(t_quant * math.log(abs(q_m))) / (2 ** (bit_width - 1) - 1)
        return d_quant

    @staticmethod
    def _quantize_helper(weight, d_quant, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output = d_quant * torch.round(input_pow.div(d_quant))
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _clip_helper(weight, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        output = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = range_pow
        output = torch.sign(weight) * output

        return output

    @staticmethod
    def _residual_helper(weight, d_quant, q_m, t_quant):
        if t_quant is None:
            t_quant = 1.0
        weight_abs = torch.abs(weight)
        q_s = 0.0
        range_pow = torch.exp(t_quant * torch.log(abs(q_m - q_s)))
        input_pow = torch.exp(t_quant * torch.log(weight_abs - q_s))  # weight_abs > q_s
        output = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        output[weight_abs <= q_s] = 0
        output[weight_abs >= q_m] = torch.round(range_pow.div(d_quant)) - range_pow.div(
            d_quant
        )
        output = torch.sign(weight) * output

        return output

    def log_qm_projection(self):
        """Log q_m during projection"""
        if (
            self.num_steps >= self.start_projection_step
            and self.num_steps <= self.start_projection_step + self.projection_steps
            and self.num_steps % 1000 == 0
        ):
            log_file = os.path.join(self.log_dir, f"projection_qm_{self.num_steps}.txt")
            with self.safe_open_file(log_file, "w") as f:
                curr_period = (
                    self.num_steps - self.start_projection_step
                ) // self.projection_period_duration
                f.write(f"Step: {self.num_steps}, Projection Period: {curr_period}\n")
                f.write(f"Current max_bit_wt: {self.max_bit_wt}\n\n")

                for group in self.param_groups:
                    for p_name, p in zip(group["p_names"], group["params"]):
                        if "q_m_wt" in p_name:
                            layer_name = ".".join(p_name.split(".")[:-1])
                            f.write(f"Layer: {layer_name}\n")
                            f.write(f"q_m stats: min={p.data.min().item():.6f}, ")
                            f.write(f"max={p.data.max().item():.6f}, ")
                            f.write(f"mean={p.data.mean().item():.6f}\n\n")

    def step(self, loss=None, closure=None):
        """
        Core function.
        """

        if closure is not None:
            _ = closure()

        self.num_steps += 1
        self.compute_grad_variant()

        # Determine the bit range projection for weights
        if (
            self.num_steps >= self.start_projection_step
            and self.num_steps <= self.start_pruning_step
            and self.start_projection_step != self.start_pruning_step
        ):
            if (
                self.num_steps - self.start_projection_step - 1
            ) % self.projection_period_duration == 0 and (
                self.num_steps - self.start_projection_step - 1
            ) != 0:
                self.max_bit_wt = self.max_bit_wt - self.bit_reduction
                self.min_bit_wt = self.min_bit_wt

        # Partition groups into important and redundant groups
        if (
            self.num_steps >= self.start_pruning_step
            and self.curr_pruning_period < self.pruning_periods
            and self.pruning_period_duration != 0
        ):
            if (
                self.num_steps - self.start_pruning_step - 1
            ) % self.pruning_period_duration == 0:
                self.logger.info(
                    f"Determining important and redundant groups using saliency scores. Step={self.num_steps}"
                )
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1

        # Second pass to update variables
        if self.pruning_period_duration != 0:
            t = (
                self.num_steps - self.start_pruning_step
            ) % self.pruning_period_duration
        for group in self.param_groups:
            if not group["is_prunable"] or len(group["active_redundant_idxes"]) == 0:
                if self.num_steps <= self.start_projection_step:  # First stage
                    # self.logger.info(
                    #     f"Warmup stage: updating trainable parameters and quantization parameters using SGD. Step={self.num_steps}"
                    # )
                    self.gradient_descent_step(group)
                elif self.num_steps > self.start_pruning_step + self.pruning_steps:
                    if (
                        self.num_steps
                        == self.start_pruning_step + self.pruning_steps + 1
                    ):
                        bit_layer = self.get_bitwidth_dict(group)
                        self.bit_layers.update(bit_layer)
                    self.partial_projected_gradient_descent_step_fix(
                        group, self.bit_layers
                    )
                else:
                    self.partial_projected_gradient_descent_step_range_wt(group)
                    # self.partial_projected_gradient_descent_step_range_act(group) # Uncomment this line if apply activation quantization
            elif (
                group["is_prunable"] and len(group["active_redundant_idxes"]) > 0
            ):  # Third stage
                # self.partial_projected_gradient_descent_step_range_act(group)
                # self.logger.info(
                #     f"Joint pruning and quantization stage. Step={self.num_steps}"
                # )
                # Add stochastic gradient term for quantization params (d_quant excluded)
                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue
                    if "t_quant_wt" in p_name or "q_m_wt" in p_name:
                        p.data.add_(
                            group["grad_variant"][p_name], alpha=-group["lr_quant"]
                        )

                # Identify redundant idxes
                active_redundant_idxes = group["active_redundant_idxes"]

                # Compute forget rate (gamma) and quant step size (d)
                gamma, d_quant = self.compute_gamma_d(
                    group, active_redundant_idxes, [self.min_bit_wt, self.max_bit_wt]
                )
                # self.logger.info(
                #     f"Forget rate (gamma): {gamma}, Quant step size (d): {d_quant}"
                # )
                self.gamma, self.d_quant = gamma, d_quant

                # Update quant step size d
                for i, (p_name, p_transform) in enumerate(
                    zip(group["p_names"], group["p_transform"])
                ):
                    if "d_quant_wt" in p_name:
                        with torch.no_grad():
                            group["params"][i].copy_(d_quant)

                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue

                    # Add redundant info removal term
                    is_quantize, quantize_weight = self.quantize_weight(group, p_name)
                    if p_transform != TensorTransform.NO_PRUNE:
                        if is_quantize:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * quantize_weight.data[active_redundant_idxes]
                            )
                        else:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * p.data[active_redundant_idxes]
                            )

                    # Add stochastic gradient term for non-quantization parameters
                    if (
                        "d_quant" not in p_name
                        and "t_quant" not in p_name
                        and "q_m" not in p_name
                    ):
                        p.data.add_(group["grad_variant"][p_name], alpha=-group["lr"])

                    # Tackle auxiliary params
                    for ng_id, offset in group["auxiliary_ngs"]:
                        active_redundant_aux_idxes = [
                            i + offset for i in active_redundant_idxes
                        ]
                        for aux_p in self.auxiliary_param_groups[ng_id]["params"]:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[active_redundant_aux_idxes, ...] *= (
                                self.pruning_period_duration - t - 1.0
                            ) / (self.pruning_period_duration - t)

            self.fix_pruned_groups_as_zeros(group)

        if self.pruning_period_duration != 0:
            if (
                self.num_steps >= self.start_pruning_step
                and t == self.pruning_period_duration - 1
            ):
                self.commit_redundant_idxes()

    def compute_metrics(self):
        """Compute optimizer metrics, skipping quantization parameters."""
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0

        for group in self.param_groups:
            if not (group["is_prunable"] and not group["is_auxiliary"]):
                continue

            norm_group = None
            import_idxes = group["important_idxes"]
            redund_idxes = group["active_redundant_idxes"] + group["pruned_idxes"]

            for param, p_transform in zip(group["params"], group["p_transform"]):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                param_transform = tensor_transformation_param_group(
                    param.data, p_transform, group
                )
                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2

            if norm_group is not None:  # Only process if we have valid parameters
                norm_group = torch.sqrt(norm_group)
                self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()
                self.opt_metrics.norm_params += torch.sum(norm_group).item()
                self.opt_metrics.norm_important_groups += torch.sum(
                    norm_group[import_idxes]
                ).item()
                self.opt_metrics.norm_redundant_groups += torch.sum(
                    norm_group[redund_idxes]
                ).item()
                self.opt_metrics.num_important_groups += len(import_idxes)
                self.opt_metrics.num_redundant_groups += len(redund_idxes)

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(
            self.total_num_groups + self.safe_guard
        )

        return self.opt_metrics

    def state_dict(self, debug=False):
        """
        Return a state_dict of the optimizer for restore.
        """
        parent_state = (
            super().state_dict()
        )  # includes param_gropups, param_data, and optimizer-specific state
        self.logger.debug(f"Parent state_dict keys: {parent_state.keys()}")
        # Add GETA quantization state
        state_dict = {"param_groups": self.param_groups}
        state_dict.update(parent_state)
        state_dict.update(
            {
                "num_steps": self.num_steps,
                "curr_pruning_period": self.curr_pruning_period,
                "start_pruning_step": self.start_pruning_step,
                "pruning_periods": self.pruning_periods,
                "pruning_steps": self.pruning_steps,
                "start_projection_step": self.start_projection_step,
                "projection_periods": self.projection_periods,
                "projection_steps": self.projection_steps,
                "pruning_period_duration": self.pruning_period_duration,
                "bit_layers": self.bit_layers,
                "projection_period_duration": self.projection_period_duration,
                "min_bit_wt": self.min_bit_wt,
                "max_bit_wt": self.max_bit_wt,
                "min_bit_act": self.min_bit_act,
                "max_bit_act": self.max_bit_act,
                "bit_reduction": self.bit_reduction,
                "pruned_group_indices": self.pruned_group_idxes,
            }
        )
        self.logger.debug(f"Final state_dict keys: {state_dict.keys()}")
        return state_dict

    # def load_state_dict(self, state_dict):
    #     """Loads the optimizer state.

    #     Args:
    #         state_dict (dict): Optimizer state dictionary containing:
    #             - param_groups: List of parameter group dictionaries
    #             - parameter data and gradients
    #             - optimizer-specific state

    #     Raises:
    #         ValueError: If state_dict missing required data, contains invalid structure,
    #             or parameters don't match current model

    #     Note:
    #         Each parameter group must contain a 'param_data' field with parameter
    #         name, data and optional gradient information. Parameter shapes must match
    #         between saved state and current model.
    #     """
    #     if "param_groups" not in state_dict:
    #         raise ValueError("Missing param_groups in state_dict")
    #     saved_groups = state_dict["param_groups"]

    #     if "num_steps" not in state_dict:
    #         raise ValueError("Missing num_steps in state_dict")
    #     self.num_steps = state_dict["num_steps"]

    #     # First, collect all parameters from saved state
    #     saved_params = {}
    #     for i, group in enumerate(saved_groups):
    #         if "param_data" not in group:
    #             raise ValueError(
    #                 f"Parameter group {i} is missing required 'param_data' field"
    #             )
    #         for param_info in group["param_data"]:
    #             if not isinstance(param_info, dict) or "name" not in param_info:
    #                 raise ValueError(
    #                     "Invalid parameter info structure in saved state, check checkpoint format"
    #                 )
    #             name = param_info["name"]
    #             if name in saved_params:
    #                 raise ValueError(
    #                     f"Duplicate parameter name '{name}' in saved state"
    #                 )
    #             saved_params[name] = param_info

    #     # Update current groups
    #     for current_group, saved_group in zip(self.param_groups, saved_groups):
    #         # Update non-parameter attributes
    #         for key in saved_group:
    #             if key not in ["params", "param_data"]:
    #                 current_group[key] = saved_group[key]

    #         # Handle parameters
    #         for current_p_name, current_param in zip(
    #             current_group["p_names"], current_group["params"]
    #         ):
    #             if current_p_name not in saved_params:
    #                 raise ValueError(
    #                     f"Parameter '{current_p_name}' not found in saved state"
    #                 )

    #             saved_param_info = saved_params[current_p_name]
    #             if "data" not in saved_param_info:
    #                 raise ValueError(f"No data found for parameter '{current_p_name}'")

    #             saved_data = saved_param_info["data"]

    #             if current_param is None:
    #                 raise ValueError(f"Current parameter '{current_p_name}' is None")
    #             if saved_data is None:
    #                 raise ValueError(
    #                     f"Saved data for parameter '{current_p_name}' is None"
    #                 )

    #             # Verify shapes match
    #             if saved_data.shape != current_param.data.shape:
    #                 raise ValueError(
    #                     f"Shape mismatch for parameter '{current_p_name}': "
    #                     f"saved={saved_data.shape}, current={current_param.data.shape}"
    #                 )
    #             # Copy data
    #             current_param.data.copy_(saved_data.to(current_param.device))
    #             if "requires_grad" in saved_param_info:
    #                 current_param.requires_grad = saved_param_info["requires_grad"]

    #             # Copy gradient if it exists
    #             if "grad_variant" in saved_param_info:
    #                 grad_data = saved_param_info["grad_variant"]
    #                 if grad_data.shape != current_param.data.shape:
    #                     raise ValueError(
    #                         f"Gradient shape mismatch for '{current_p_name}': "
    #                         f"saved={grad_data.shape}, param={current_param.data.shape}"
    #                     )
    #                 if "grad_variant" not in current_group:
    #                     current_group["grad_variant"] = {}
    #                 grad_data = grad_data.to(current_param.device)
    #                 if current_p_name in current_group["grad_variant"]:
    #                     current_group["grad_variant"][current_p_name].copy_(grad_data)
    #                 else:
    #                     current_group["grad_variant"][current_p_name] = grad_data

    #     # Rebuild auxiliary param groups
    #     self.auxiliary_param_groups = {
    #         group["id"]: group
    #         for group in self.param_groups
    #         if group.get("is_auxiliary", False)
    #     }

    #     del state_dict
    def load_state_dict(self, state_dict):
        import copy

        for attr_name in state_dict:
            if attr_name != "param_groups":
                setattr(self, attr_name, state_dict[attr_name])
            else:
                prev_param_groups = state_dict[attr_name]
                for param_group in self.param_groups:
                    prev_param_group = next(
                        (
                            _g
                            for _g in prev_param_groups
                            if param_group["id"] == _g["id"]
                        ),
                        None,
                    )
                    if prev_param_group is None:
                        raise Warning(
                            f"Param group {param_group['id']} not found in previous state_dict."
                        )
                        continue
                    for param_group_attr_name in prev_param_group:
                        if param_group_attr_name == "params":
                            for p_name, param, prev_p_name, prev_param in zip(
                                param_group["p_names"],
                                param_group["params"],
                                prev_param_group["p_names"],
                                prev_param_group["params"],
                            ):
                                if p_name == prev_p_name:
                                    param.data.copy_(prev_param.data)
                                    if prev_param.grad is not None:
                                        param.grad.copy_(prev_param.grad)
                                else:
                                    print(
                                        f"\tParam {p_name} not found in previous state_dict."
                                    )
                        else:
                            param_group[param_group_attr_name] = copy.deepcopy(
                                prev_param_group[param_group_attr_name]
                            )

        self.auxiliary_param_groups = {
            group["id"]: group
            for group in self.param_groups
            if group.get("is_auxiliary", False)
        }

        del state_dict

    def create_checkpoint(self, model, epoch, loss):
        """Creates a standardized checkpoint dictionary.

        Returns:
            dict: A checkpoint containing model and optimizer state, plus metadata.
        """
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.state_dict(),  # Contains num_steps
            "epoch": epoch,
            "loss": loss,
        }
