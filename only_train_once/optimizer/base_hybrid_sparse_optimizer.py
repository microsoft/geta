import numpy as np
import torch
from torch.optim.optimizer import required

from only_train_once.transform import (
    TensorTransform,
    index_transformation_param_group,
    tensor_transformation_param_group,
)

from .base_optimizer import BaseOptimizer
from .hyperparameter import DEFAULT_OPT_PARAMS, SUPPORT_GRADIENT_ESTIMATES
from .importance_score import calculate_importance_score


class SparseOptimizerMetrics:
    num_groups = 0
    num_zero_groups = 0
    num_important_groups = 0
    num_redundant_groups = 0

    # For CRIC
    num_violating_groups = 0
    num_trial_violating_groups = 0
    num_historical_violating_groups = 0

    norm_violating_groups = 0.0

    norm_params = 0.0
    norm_important_groups = 0.0
    norm_redundant_groups = 0.0

    group_sparsity = 0.0

    def __repr__(self) -> str:
        return f"num_zero_grps: {self.num_zero_groups}, gs: {self.group_sparsity:.2f}, norm_params: {self.norm_params:.2f}, norm_import: {self.norm_important_groups:.2f}, norm_violating: {self.norm_violating_groups:.2f}, norm_redund: {self.norm_redundant_groups:.2f}, num_grps_import: {self.num_important_groups}, num_grps_redund: {self.num_redundant_groups}, num_grps_violating: {self.num_violating_groups}, num_grps_trial_violating: {self.num_trial_violating_groups}, num_grps_hist_violating: {self.num_historical_violating_groups}"


class BaseHybridSparseOptimizer(BaseOptimizer):
    def __init__(
        self,
        params,
        variant="sgd",
        lr=required,
        first_momentum=None,
        second_momentum=None,
        dampening=None,
        weight_decay=None,
        target_group_sparsity=0.0,
        group_divisible=1,
        additional_defaults=dict(),
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if variant not in SUPPORT_GRADIENT_ESTIMATES:
            raise ValueError(
                f"Need to select a gradient estimation from {SUPPORT_GRADIENT_ESTIMATES}"
            )

        # Set up hyper-parameters related to baseline optimizer
        first_momentum = (
            first_momentum
            if first_momentum is not None
            else DEFAULT_OPT_PARAMS[variant]["first_momentum"]
        )
        second_momentum = (
            second_momentum
            if second_momentum is not None
            else DEFAULT_OPT_PARAMS[variant]["second_momentum"]
        )
        dampening = (
            dampening
            if dampening is not None
            else DEFAULT_OPT_PARAMS[variant]["dampening"]
        )
        weight_decay = (
            weight_decay
            if weight_decay is not None
            else DEFAULT_OPT_PARAMS[variant]["weight_decay"]
        )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            variant=variant,
            grad_variant=dict(),
            global_start_idx=0,
            global_idx=0,
        )
        defaults.update(additional_defaults)

        super().__init__(params, defaults)

        # Set up total number of prunable groups
        self.total_num_groups = 0
        for param_group in params:
            if param_group["is_prunable"] and not param_group["is_auxiliary"]:
                if param_group["num_groups"] <= group_divisible:
                    param_group["is_prunable"] = False
                else:
                    self.total_num_groups += param_group["num_groups"]

        self.group_divisible = group_divisible
        self.target_group_sparsity = target_group_sparsity
        self.target_num_redundant_groups = int(
            self.total_num_groups * min(self.target_group_sparsity, 0.999)
        )
        self.opt_metrics = SparseOptimizerMetrics()

        self.auxiliary_param_groups = dict()
        for group in self.param_groups:
            if group["is_auxiliary"]:
                self.auxiliary_param_groups[group["id"]] = group

    def gradient_descent_step(self, param_group):
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                p.data.add_(
                    param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                )

            p.data.add_(param_group["grad_variant"][p_name], alpha=-param_group["lr"])

    def fix_pruned_groups_as_zeros(self, param_group):
        if len(param_group["pruned_idxes"]) > 0:
            for p, p_transform in zip(
                param_group["params"], param_group["p_transform"]
            ):
                pruned_idxes = index_transformation_param_group(
                    param_group["pruned_idxes"], p_transform, param_group
                )
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                else:
                    if (
                        p_transform == TensorTransform.TRANSPOSE
                        and len(p.data.shape) > 1
                    ):
                        p.data[:, pruned_idxes] = 0.0
                    else:
                        p.data[pruned_idxes] = 0.0

            # Tackle auxiliary params
            for ng_id, offset in param_group["auxiliary_ngs"]:
                pruned_aux_idxes = [i + offset for i in pruned_idxes]
                for aux_p in self.auxiliary_param_groups[ng_id]["params"]:
                    if aux_p.grad is None:
                        continue
                    aux_p.data[pruned_aux_idxes, ...] = 0.0

    def compute_importance_scores(self, **kwargs):
        global_start_idx = 0
        self.global_scores = list()  # Accumulate global scores
        # Calculate raw importance scores by varying criteria
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                calculate_importance_score(self.importance_score_criteria, group)

        # Normalize importance_score
        # Calculate normalization_denoms
        normalization_denoms = dict.fromkeys(
            self.importance_score_criteria.keys(), self.safe_guard
        )
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                for proxy_name in self.importance_score_criteria:
                    if not proxy_name in group["importance_scores"]:
                        continue
                    normalization_denoms[proxy_name] += torch.sum(
                        group["importance_scores"][proxy_name] ** 2, dim=0
                    ).item()
        for proxy_name in normalization_denoms:
            normalization_denoms[proxy_name] = (
                np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard
            )

        global_start_idx = 0
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                group["importance_scores"]["overall"] = None
                for proxy_name in self.importance_score_criteria:
                    if not proxy_name in group["importance_scores"]:
                        continue
                    group["importance_scores"][proxy_name].mul_(
                        self.importance_score_criteria[proxy_name]
                        / normalization_denoms[proxy_name]
                    )
                    if group["importance_scores"]["overall"] is None:
                        group["importance_scores"]["overall"] = group[
                            "importance_scores"
                        ][proxy_name].clone()
                    else:
                        group["importance_scores"]["overall"] += group[
                            "importance_scores"
                        ][proxy_name]
                group["global_start_idx"] = global_start_idx
                group["global_idxes"] = np.arange(
                    global_start_idx, global_start_idx + group["num_groups"]
                )
                global_start_idx += group["num_groups"]
                self.global_scores.append(group["importance_scores"]["overall"])
        num_count = 0
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                num_count += 1

    def compute_metrics(self):
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
                """
                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'], group['num_heads'])
                elif isinstance(p_transform, list):
                    param_transform = param.data.clone()
                    for (p_transform_type, p_transform_config) in p_transform:
                        if p_transform_type == TensorTransform.MULTIHEAD_HEADDIM:
                            head_dim = p_transform_config['head_dim']
                            num_heads = p_transform_config['num_heads']
                            param_transform = tensor_transformation(param_transform, p_transform_type, num_groups=head_dim, num_heads=num_heads)
                        elif p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD or p_transform_type == TensorTransform.MULTIHEAD_NUMHEAD_SPREAD:
                            num_heads = p_transform_config['num_heads']
                            param_transform = tensor_transformation(param_transform, p_transform_type, num_heads)
                else:
                    param_transform = tensor_transformation(param.data, p_transform, group['num_groups'])
                """
                param_transform = tensor_transformation_param_group(
                    param.data, p_transform, group
                )
                if norm_group == None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
                # if p_transform == TensorTransform.NO_PRUNE:
                #     continue
                # param_transform = None
                # if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                #     param_transform = tensor_transformation_param_group(
                #         param.data, p_transform, group["num_groups"], group["num_heads"]
                #     )
                # else:
                #     param_transform = tensor_transformation_param_group(
                #         param.data, p_transform, group["num_groups"]
                #     )
                # if norm_group == None:
                #     norm_group = torch.norm(param_transform, dim=1) ** 2
                # else:
                #     norm_group += torch.norm(param_transform, dim=1) ** 2
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

    def state_dict(self):
        """
        Return a state_dict of the optimizer for restore.
        """
        state_dict = super().state_dict()

        state_dict["group_divisible"] = self.group_divisible
        state_dict["target_group_sparsity"] = self.target_group_sparsity
        state_dict["total_num_groups"] = self.total_num_groups
        state_dict["target_num_redundant_groups"] = self.target_num_redundant_groups

        return state_dict

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
                            f"Param group {param_group['id']} does not find in previous state_dict."
                        )
                        continue
                    for param_group_attr_name in prev_param_group:
                        if param_group_attr_name == "params":
                            # Params need in-place inheritance.
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
                                        f"\tParam {p_name} does not find in previous state_dict."
                                    )
                        else:
                            param_group[param_group_attr_name] = copy.deepcopy(
                                prev_param_group[param_group_attr_name]
                            )

        self.auxiliary_param_groups = dict()
        for group in self.param_groups:
            if group["is_auxiliary"]:
                self.auxiliary_param_groups[group["id"]] = group

        del state_dict
