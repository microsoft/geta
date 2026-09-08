import torch

from only_train_once.transform import (
    TensorTransform,
    tensor_transformation,
    tensor_transformation_param_group,
)

LORA_NAMES = [("lora_B", "lora_A"), ("lora_embedding_B", "lora_embedding_A")]


def importance_score_by_first_order_taylor(param_group):
    params_grads_inner_prod = None
    for p_name, param, p_transform in zip(
        param_group["p_names"], param_group["params"], param_group["p_transform"]
    ):
        if p_name not in param_group["grad_variant"]:
            continue
        param_transform = tensor_transformation_param_group(
            param.data, p_transform, param_group
        )
        grad = param_group["grad_variant"][p_name]
        grad_transform = tensor_transformation_param_group(
            grad, p_transform, param_group
        )

        if p_transform == TensorTransform.NO_PRUNE:
            continue
        else:
            if params_grads_inner_prod == None:
                params_grads_inner_prod = torch.sum(
                    param_transform * grad_transform, dim=1
                )
            else:
                params_grads_inner_prod += torch.sum(
                    param_transform * grad_transform, dim=1
                )
    param_group["importance_scores"]["taylor_first_order"] = torch.abs(
        params_grads_inner_prod
    )


def importance_score_by_second_order_taylor(param_group):
    if "taylor_first_order" in param_group["importance_scores"]:
        param_group["importance_scores"]["taylor_second_order"] = (
            0.5 * param_group["importance_scores"]["taylor_first_order"] ** 2
        )
        return

    params_grads_inner_prod = None
    for p_name, param, p_transform in zip(
        param_group["p_names"], param_group["params"], param_group["p_transform"]
    ):
        if p_name not in param_group["grad_variant"]:
            continue
        param_transform = tensor_transformation_param_group(
            param.data, p_transform, param_group
        )
        grad = param_group["grad_variant"][p_name]
        grad_transform = tensor_transformation_param_group(
            grad, p_transform, param_group
        )
        if params_grads_inner_prod == None:
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            params_grads_inner_prod += torch.sum(
                param_transform * grad_transform, dim=1
            )
    param_group["importance_scores"]["taylor_second_order"] = (
        0.5 * params_grads_inner_prod**2
    )


def importance_score_by_first_order_taylor_lora(param_group, global_params):
    params_grads_inner_prod = None
    for p_name, param, p_transform in zip(
        param_group["p_names"], param_group["params"], param_group["p_transform"]
    ):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + "weight"
                original_param = global_params[original_param_name]

                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(
                        original_param,
                        p_transform,
                        param_group["num_groups"],
                        param_group["num_heads"],
                    )
                elif lora_strs[0] == "lora_embedding_B":
                    param_transform = tensor_transformation(
                        original_param,
                        TensorTransform.TRANSPOSE,
                        param_group["num_groups"],
                    )
                else:
                    param_transform = tensor_transformation(
                        original_param, p_transform, param_group["num_groups"]
                    )
                grad_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    grad_transform = tensor_transformation(
                        lora_BA,
                        p_transform,
                        param_group["num_groups"],
                        param_group["num_heads"],
                    )
                else:
                    grad_transform = tensor_transformation(
                        lora_BA, p_transform, param_group["num_groups"]
                    )

                if params_grads_inner_prod == None:
                    params_grads_inner_prod = torch.sum(
                        param_transform * grad_transform, dim=1
                    )
                else:
                    params_grads_inner_prod += torch.sum(
                        param_transform * grad_transform, dim=1
                    )

    param_group["importance_scores"]["taylor_first_order"] = torch.abs(
        params_grads_inner_prod
    )


def importance_score_by_second_order_taylor_lora(param_group, global_params):
    if "taylor_first_order" in param_group["importance_scores"]:
        param_group["importance_scores"]["taylor_second_order"] = (
            0.5 * param_group["importance_scores"]["taylor_first_order"] ** 2
        )
        return

    params_grads_inner_prod = None
    for p_name, param, p_transform in zip(
        param_group["p_names"], param_group["params"], param_group["p_transform"]
    ):
        for lora_strs in LORA_NAMES:
            if lora_strs[0] in p_name:
                lora_A_name = p_name.replace(lora_strs[0], lora_strs[1])
                lora_A = global_params[lora_A_name]
                lora_BA = torch.matmul(param, lora_A)
                original_param_name = p_name.split(lora_strs[0])[0] + "weight"
                original_param = global_params[original_param_name]

                param_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    param_transform = tensor_transformation(
                        original_param,
                        p_transform,
                        param_group["num_groups"],
                        param_group["num_heads"],
                    )
                elif lora_strs[0] == "lora_embedding_B":
                    param_transform = tensor_transformation(
                        original_param,
                        TensorTransform.TRANSPOSE,
                        param_group["num_groups"],
                    )
                else:
                    param_transform = tensor_transformation(
                        original_param, p_transform, param_group["num_groups"]
                    )
                grad_transform = None
                if p_transform == TensorTransform.MULTIHEAD_HEADDIM:
                    grad_transform = tensor_transformation(
                        lora_BA,
                        p_transform,
                        param_group["num_groups"],
                        param_group["num_heads"],
                    )
                else:
                    grad_transform = tensor_transformation(
                        lora_BA, p_transform, param_group["num_groups"]
                    )

                if params_grads_inner_prod == None:
                    params_grads_inner_prod = torch.sum(
                        param_transform * grad_transform, dim=1
                    )
                else:
                    params_grads_inner_prod += torch.sum(
                        param_transform * grad_transform, dim=1
                    )

    param_group["importance_scores"]["taylor_second_order"] = (
        0.5 * params_grads_inner_prod**2
    )
