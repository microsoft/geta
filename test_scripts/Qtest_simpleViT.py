"""
Debug script
"""

import argparse
import json
import logging
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# from transformers import AutoImageProcessor
from only_train_once import OTO
from only_train_once.optimizer.utils import (
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.simple_vit import simpleViT_cifar10
from test_scripts.geta_common import (
    add_common_args,
    resolve_data_dir,
    resolve_output_dir,
)
from utils.utils import check_accuracy

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger("new")


def get_quant_param_dict(model):
    # Access quantization parameter information
    param_dict = {}
    for name, param in model.named_parameters():
        if "d_quant" in name or "t_quant" in name or "q_m" in name:
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            if layer_name in param_dict:
                param_dict[layer_name][param_name] = param.item()
            else:
                param_dict[layer_name] = {}
                param_dict[layer_name][param_name] = param.item()
    return param_dict


def get_bitwidth_dict(param_dict):
    bit_dict = {}

    for key in param_dict.keys():
        bit_dict[key] = {}

        d_quant_wt = param_dict[key]["d_quant_wt"]
        q_m_wt = abs(param_dict[key]["q_m_wt"])
        if "t_quant_wt" in param_dict[key]:
            t_quant_wt = param_dict[key]["t_quant_wt"]
        else:
            t_quant_wt = 1.0
        bit_width_wt = (
            math.log2(math.exp(t_quant_wt * math.log(q_m_wt)) / abs(d_quant_wt) + 1) + 1
        )
        bit_dict[key]["weight"] = bit_width_wt

        if "d_quant_act" in param_dict[key]:
            d_quant_act = param_dict[key]["d_quant_act"]
            q_m_act = abs(param_dict[key]["q_m_act"])
            if "t_quant_act" in param_dict[key]:
                t_quant_act = param_dict[key]["t_quant_act"]
            else:
                t_quant_act = 1.0
            bit_width_act = (
                math.log2(
                    math.exp(t_quant_act * math.log(q_m_act)) / abs(d_quant_act) + 1
                )
                + 1
            )
            bit_dict[key]["activation"] = bit_width_act

    return bit_dict


def get_data_loader(dataset: str, batch_size: int, num_workers: int, data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    if dataset == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        trainset = CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root=os.path.join(data_dir, "cifar10"),
            train=False,
            download=True,
            transform=transform_test,
        )
        input_size = (1, 3, 32, 32)
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        raise ValueError("Unsupported dataset")

    return train_loader, test_loader, input_size


def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y


def cross_entropy_onehot_target(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def mixup_func(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(
        1 - gamma, perm_target
    )


def main(config):
    model_name = config.model_name
    dataset = config.dataset
    init_bit = config.init_bit
    batch_size = config.batch_size
    num_workers = config.num_workers
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    sparsity_level = config.sparsity_level
    projection_start_step = config.projection_start_step
    projection_periods = config.projection_periods
    projection_steps = config.projection_steps
    pruning_start_step = config.pruning_start_step
    pruning_periods = config.pruning_periods
    pruning_steps = config.pruning_steps
    lr_step = config.lr_step
    lr_gamma = config.lr_gamma
    variant = config.variant
    bit_reduction = config.bit_reduction
    min_bit = config.min_bit
    max_bit = config.max_bit
    mix_up = config.mix_up
    label_smooth = config.label_smooth
    seed = config.seed

    assert pruning_start_step == projection_start_step + projection_steps
    output_dir = resolve_output_dir(
        config.output_dir, f"{model_name}_{variant}_{sparsity_level}"
    )
    data_dir = resolve_data_dir(config.data_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Logging configuration
    logging.basicConfig(
        filename=os.path.join(
            output_dir,
            f"{model_name}_{variant}_{sparsity_level}_{pruning_start_step}.txt",
        ),
        filemode="a",
        format="%(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Setup info
    logger.info(f"Model name: {model_name:^s}")
    logger.info(f"Total epochs: {epochs:^4d}")
    logger.info(f"Sparsity level: {sparsity_level:^.2f}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Optimizer variant: {variant:^s}")
    logger.info(f"Start projection step: {projection_start_step:^3d}")
    logger.info(f"Projection steps: {projection_steps:^3d}")
    logger.info(f"Start pruning step: {pruning_start_step:^3d}")
    logger.info(f"Pruning steps: {pruning_steps:^3d}")
    logger.info(f"Learning rate scheduler steps: {lr_step:^3d}")
    logger.info("=======================================")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = 1
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    train_loader, test_loader, input_size = get_data_loader(
        dataset, batch_size * num_gpus, num_workers, data_dir
    )
    # num_classes = 10 if dataset == "cifar10" else 1000
    dummy_input = torch.rand(input_size).to(device)

    model = simpleViT_cifar10()
    q_model = model_to_quantize_model(model, num_bits=init_bit)
    oto = OTO(q_model.to(device), dummy_input=dummy_input)
    oto.mark_unprunable_by_param_names(["to_patch_embedding.2.weight"])

    # Add the visualization to make sure that everything quant_act_layers.py works well.
    # oto.visualize(view=False, out_dir='./cache', display_flops=True, display_params=True, display_macs=True)
    # exit()

    optimizer = oto.geta(
        variant=variant,
        lr=learning_rate,
        lr_quant=1e-4,
        first_momentum=0.9,
        weight_decay=weight_decay,
        target_group_sparsity=sparsity_level,
        start_projection_step=projection_start_step * len(train_loader),
        projection_periods=projection_periods,
        projection_steps=projection_steps * len(train_loader),
        start_pruning_step=pruning_start_step * len(train_loader),
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps * len(train_loader),
        bit_reduction=bit_reduction,
        min_bit_wt=min_bit,
        max_bit_wt=max_bit,
        # min_bit_act=min_bit_act,
        # max_bit_act=max_bit_act,
    )

    # Get full/original floating-point model MACs, BOPs, and number of parameters
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    full_weight_size = oto.compute_weight_size(in_million=True)

    # hard fix for full_bops calculation
    full_bops["total"] = full_bops["total"] * 32 / init_bit

    if not label_smooth:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = cross_entropy_onehot_target
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma
    )
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)

    best_epoch = 0
    best_acc1 = 0.0
    loss_list = []
    start_epoch = 0
    if (
        os.environ.get("TRAINER_RESUME") == "1"
        or os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        ckpt_path = scan_checkpoint(checkpoint_dir, "ckpt_")
        if ckpt_path is not None:
            try:
                logger.info(f"Attempting resume from {ckpt_path}")
                ckpt = load_checkpoint(ckpt_path, device)
                model_to_load = model.module if num_gpus > 1 else model
                model_to_load.load_state_dict(ckpt["model_state_dict"])
                opt_state = ckpt.get("optimizer_state_dict", {})
                for key in [
                    "num_steps",
                    "curr_pruning_period",
                    "start_pruning_step",
                    "pruning_periods",
                    "pruning_steps",
                    "start_projection_step",
                    "projection_periods",
                    "projection_steps",
                    "pruning_period_duration",
                    "projection_period_duration",
                    "target_num_redundant_groups",
                    "pruned_group_idxes",
                    "bit_layers",
                    "min_bit_wt",
                    "max_bit_wt",
                    "min_bit_act",
                    "max_bit_act",
                ]:
                    if key in opt_state:
                        try:
                            setattr(optimizer, key, opt_state[key])
                        except Exception:
                            pass
                try:
                    ckpt_groups = opt_state.get("param_groups", [])
                    for pg, ckpt_pg in zip(optimizer.param_groups, ckpt_groups):
                        for k in [
                            "important_idxes",
                            "active_redundant_idxes",
                            "pruned_idxes",
                            "importance_scores",
                        ]:
                            if k in ckpt_pg:
                                pg[k] = ckpt_pg[k]
                except Exception as e:
                    logger.warning(f"Could not restore param_groups: {e}")
                if (
                    "scheduler_state_dict" in ckpt
                    and ckpt["scheduler_state_dict"] is not None
                ):
                    try:
                        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception as e:
                        logger.warning(f"Could not load scheduler state: {e}")
                start_epoch = ckpt["epoch"] + 1
                best_acc1 = ckpt.get("best_acc1", 0.0)
                best_epoch = ckpt.get("best_epoch", 0)
                logger.info(
                    f"Resumed from epoch {start_epoch} (ckpt epoch {ckpt['epoch']}), best_acc1={best_acc1:.2f}%"
                )
            except Exception as e:
                logger.warning(f"Failed to resume from checkpoint {ckpt_path}: {e}")
                import traceback

                logger.warning(traceback.format_exc())
                start_epoch = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            if dataset == "imagenet":
                inputs, targets = batch["pixel_values"], batch["labels"]
            else:
                inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # with torch.no_grad():
            #     if label_smooth and not mix_up:
            #         targets = one_hot(targets, num_classes=10, smoothing_eps=0.1)
            #     if not label_smooth and mix_up:
            #         targets = one_hot(targets, num_classes=10)
            #         inputs, targets = mixup_func(inputs, targets)
            #     if mix_up and label_smooth:
            #         targets = one_hot(targets, num_classes=10, smoothing_eps=0.1)
            #         inputs, targets = mixup_func(inputs, targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            running_loss += loss.item()

            # if batch_idx == 70:
            #     break

            # Parameter information
            # with open("./log_new/param_info.txt", "a") as f1:
            #     f1.write(f"Epoch: {epoch}, Batch:{batch_idx}\n")
            # opt_metrics_epoch = optimizer.compute_metrics()

        lr_scheduler.step()

        opt_metrics = optimizer.compute_metrics()
        running_loss_avg = running_loss / len(train_loader)

        accuracy1, accuracy5 = check_accuracy(
            model.module if num_gpus > 1 else model, test_loader, two_input=False
        )
        avg_wt_bit = oto.compute_average_bit_width()
        logger.info(
            f"Epoch: {epoch}, loss: {running_loss_avg:5.3f}, norm_all: {opt_metrics.norm_params:5.2f}, grp_sparsity: {opt_metrics.group_sparsity:5.2f}, acc1: {accuracy1:5.2f}%, acc5: {accuracy5:5.2f}%, norm_import: {opt_metrics.norm_important_groups:5.2f}, norm_redund: {opt_metrics.norm_redundant_groups:5.2f}, num_grp_import: {opt_metrics.num_important_groups:5.2f}, num_grp_redund: {opt_metrics.num_redundant_groups:5.2f}, avg_wt_bit_width: {avg_wt_bit:5.2f}"
        )

        if accuracy1 > best_acc1:
            best_acc1 = accuracy1
            best_epoch = epoch
            torch.save(model, os.path.join(output_dir, "simpleViT_best_acc1.pt"))
        # Save checkpoint for resume (every epoch)
        try:
            ckpt = optimizer.create_checkpoint(
                model.module if num_gpus > 1 else model, epoch, running_loss_avg
            )
            ckpt["best_acc1"] = best_acc1
            ckpt["best_epoch"] = best_epoch
            try:
                ckpt["scheduler_state_dict"] = lr_scheduler.state_dict()
            except:
                ckpt["scheduler_state_dict"] = None
            save_checkpoint(os.path.join(checkpoint_dir, f"ckpt_{epoch}.pt"), ckpt)
            ckpts = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_")],
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            for old in ckpts[:-3]:
                try:
                    os.remove(os.path.join(checkpoint_dir, old))
                except:
                    pass
        except Exception as e:
            logger.warning(f"Failed to save checkpoint at epoch {epoch}: {e}")

        loss_list.append(running_loss_avg)

    logger.info(f"Best epoch: {best_epoch}. Best acc1: {best_acc1}%")
    logger.info("Training completed. Constructing subnet...")

    # Construct the subnet and get the compressed model
    oto.construct_subnet(out_dir=os.path.join(output_dir, "subnet"))
    compressed_model = torch.load(oto.compressed_model_path)
    oto_compressed = OTO(compressed_model, dummy_input)

    logger.info(f"Full MACs for Q{model_name}: {full_macs['total']} M MACs")
    logger.info(f"Full BOPs for Q{model_name}: {full_bops['total']} M BOPs")
    logger.info(f"Full num params for Q{model_name}: {full_num_params} M params")
    logger.info(f"Full weight size for Q{model_name}: {full_weight_size['total']} MB")
    if "layer_info" in full_macs and "layer_info" in full_bops:
        logger.info("Layer-by-layer breakdown for full model:")
        logger.info(f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}")
        logger.info("-" * 75)
        for mac_info, bop_info in zip(full_macs["layer_info"], full_bops["layer_info"]):
            logger.info(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    # Get compressed model MACs, BOPs, and number of parameters
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
    compressed_num_params = oto_compressed.compute_num_params(in_million=True)
    compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)

    logger.info(f"Compressed MACs for Q{model_name}: {compressed_macs['total']} M MACs")
    logger.info(f"Compressed BOPs for Q{model_name}: {compressed_bops['total']} M BOPs")
    logger.info(
        f"Compressed num params for Q{model_name}: {compressed_num_params} M params"
    )
    logger.info(
        f"Compressed weight size for Q{model_name}: {compressed_weight_size['total']} MB"
    )
    if "layer_info" in compressed_macs and "layer_info" in compressed_bops:
        logger.info("Layer-by-layer breakdown for compressed model:")
        logger.info(f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}")
        logger.info("-" * 75)
        for mac_info, bop_info in zip(
            compressed_macs["layer_info"], compressed_bops["layer_info"]
        ):
            logger.info(
                f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
            )

    logger.info(
        f"MAC reduction    : {(1.0 - compressed_macs['total'] / full_macs['total']) * 100}%"
    )
    logger.info(
        f"BOP reduction    : {(1.0 - compressed_bops['total'] / full_bops['total']) * 100}%"
    )
    logger.info(
        f"Param reduction  : {(1.0 - compressed_num_params / full_num_params) * 100}%"
    )
    logger.info(f"MAC ratio: {full_macs['total'] / compressed_macs['total']}")
    logger.info(
        f"BOP compresion ratio: {full_bops['total'] / compressed_bops['total']}"
    )

    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024**3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024**3)
    logger.info(f"Size of full/ model: {full_model_size:.4f} GB")
    logger.info(f"Size of compressed model: {compressed_model_size:.4f} GB")

    # Print and visualize each layer bit width info
    param_dict = get_quant_param_dict(model)
    bit_dict = get_bitwidth_dict(param_dict)
    logger.info("=========================")
    logger.info(json.dumps(bit_dict))


def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="simpleViT",
        help="Model architecture to use (resnet56)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset to use (cifar10 or imagenet)",
    )
    parser.add_argument(
        "--init_bit", type=int, default=16, help="Initial bit width to be used"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--sparsity_level", type=float, default=0.3, help="Sparsity Level"
    )
    parser.add_argument(
        "--projection_start_step", type=int, default=5, help="Start projection step"
    )
    parser.add_argument(
        "--projection_periods", type=int, default=5, help="Number of projection periods"
    )
    parser.add_argument(
        "--projection_steps", type=int, default=5, help="Number of projection steps"
    )
    parser.add_argument(
        "--pruning_start_step", type=int, default=10, help="Start pruning step"
    )
    parser.add_argument(
        "--pruning_periods", type=int, default=10, help="Number of pruning periods"
    )
    parser.add_argument(
        "--pruning_steps", type=int, default=20, help="Number of pruning steps"
    )
    parser.add_argument(
        "--lr_step", type=int, default=100, help="LR scheduler step size"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.1, help="LR scheduler gamma"
    )
    parser.add_argument("--variant", type=str, default="adam", help="Method variant")
    parser.add_argument(
        "--bit_reduction",
        type=int,
        default=2,
        help="bit width range upper bound decrease each step",
    )
    parser.add_argument(
        "--min_bit", type=int, default=4, help="bit width range minimum"
    )
    parser.add_argument(
        "--max_bit", type=int, default=16, help="bit width range maximum"
    )
    parser.add_argument("--mix_up", type=int, default=0, help="mixup yes(1) or no(0)")
    parser.add_argument(
        "--label_smooth", type=int, default=0, help="label smoothing yes(1) or no(0)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Parse arguments
    config = add_common_args(parser).parse_args()

    return config


if __name__ == "__main__":
    main(get_config())
