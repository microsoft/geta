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
import torchvision
from torch import distributed, nn

# from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from test_scripts.geta_common import (
    add_common_args,
    create_exp_dir,
    resolve_data_dir,
)
from utils.utils import check_accuracy

# Ignore warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger("new")


def get_dist_info(args):
    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.num_gpus = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl", rank=args.rank, world_size=args.num_gpus)
    except KeyError:
        args.rank = 0
        args.local_rank = 0
        args.num_gpus = 1
    return args


def prepare_dist_model(model, args):
    model = model.cpu()
    torch.cuda.set_device(args.local_rank)
    torch.cuda.empty_cache()
    model = model.to(torch.cuda.current_device())
    if args.num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, broadcast_buffers=False, device_ids=[args.local_rank]
        )
    return model


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


def get_data_loader(dataset: str, batch_size: int, num_workers: int, args: None):
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
            root=os.path.join(resolve_data_dir(), "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root=os.path.join(resolve_data_dir(), "cifar10"),
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
    elif dataset == "imagenet":
        input_size = (1, 3, 224, 224)
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_set = torchvision.datasets.ImageFolder(
            root=args.train_dir, transform=transform_train
        )
        test_set = torchvision.datasets.ImageFolder(
            root=args.test_dir, transform=transform_test
        )

        if args.ddp:
            train_sampler = DistributedSampler(train_set)
            val_sampler = DistributedSampler(test_set, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=8,
            pin_memory=True,
            sampler=train_sampler,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            sampler=val_sampler,
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
    lr = config.lr
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
    min_bit_wt = config.min_bit_wt
    max_bit_wt = config.max_bit_wt
    min_bit_act = config.min_bit_act
    max_bit_act = config.max_bit_act
    mix_up = config.mix_up
    label_smooth = config.label_smooth
    seed = config.seed

    print(config)
    assert pruning_start_step == projection_start_step + projection_steps
    config = get_dist_info(config)
    # Logging configuration
    # logging.basicConfig(
    #     filename=f"outputs/{model_name}_{variant}_{sparsity_level}_{pruning_start_step}.txt",
    #     filemode="a",
    #     format="%(message)s",
    #     level=logging.INFO,
    # )
    # logger = logging.getLogger(__name__)
    logger = create_exp_dir(config, "outputs", "qvit_imagenet")
    output_dir = os.path.dirname(logger.handlers[0].baseFilename)

    # Setup info
    logger.info(f"Model name: {model_name:^s}")
    logger.info(f"Total epochs: {epochs:^4d}")
    logger.info(f"Sparsity level: {sparsity_level:^.2f}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Optimizer variant: {variant:^s}")
    logger.info(f"Start projection step: {projection_start_step:^3d}")
    logger.info(f"Projection steps: {projection_steps:^3d}")
    logger.info(f"Start pruning step: {pruning_start_step:^3d}")
    logger.info(f"Pruning steps: {pruning_steps:^3d}")
    logger.info(f"Learning rate scheduler steps: {lr_step:^3d}")
    logger.info("=======================================")

    torch.manual_seed(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.cuda.current_device()
    device = torch.device(f"cuda:{config.local_rank}")
    torch.cuda.manual_seed(seed)

    if config.ddp:
        batch_size = batch_size // config.num_gpus

    train_loader, test_loader, input_size = get_data_loader(
        dataset, batch_size, num_workers, config
    )

    dummy_input = torch.rand(input_size)

    if model_name == "vit":
        from sanity_check.backends.vision_transformer.vision_transformer import (
            vit_small_patch16_224,
        )

        model = vit_small_patch16_224(pretrained=True, num_classes=1000)
    elif model_name == "deit":
        from sanity_check.backends.vision_transformer.DeiT import deit_tiny_patch16_224

        model = deit_tiny_patch16_224(pretrained=True, num_classes=1000)
    elif model_name == "pvt":
        from sanity_check.backends.vision_transformer.PVT import pvt_v2_b0

        model = pvt_v2_b0(pretrained=True, num_classes=1000)
    elif model_name == "swin":
        from sanity_check.backends.vision_transformer.Swin import (
            swin_tiny_patch4_window7_224,
        )

        model = swin_tiny_patch4_window7_224(pretrained=True, num_classes=1000)

    accuracy1, accuracy5 = check_accuracy(
        model.to(device), test_loader, two_input=False
    )
    logger.info(
        f"Initial accuracy before quantization: {accuracy1:5.2f}, accuract5: {accuracy5:5.2f}%"
    )

    model = model_to_quantize_model(model, num_bits=init_bit)
    oto = OTO(model.cpu(), dummy_input=dummy_input.cpu())

    if model_name == "vit" or model_name == "deit":
        oto.mark_unprunable_by_param_names(["patch_embed.proj.weight", "pos_embed"])
    elif model_name == "pvt":
        model = None
    elif model_name == "swin":
        unprunable_list = ["patch_embed.proj.weight", "pos_embed"]
        for name, param in model.named_parameters():
            if "attn.qkv." in name:
                unprunable_list.append(name)

        oto.mark_unprunable_by_param_names(unprunable_list)

    if config.ddp:
        model = prepare_dist_model(model, config)
    else:
        model = model.cuda(device)

    accuracy1, accuracy5 = check_accuracy(
        model.module if config.num_gpus > 1 else model, test_loader, two_input=False
    )
    logger.info(f"Initial accuracy: {accuracy1:5.2f}, accuract5: {accuracy5:5.2f}%")
    # Add the visualization to make sure that everything quant_act_layers.py works well.
    # oto.visualize(view=False, out_dir='./cache', display_flops=True, display_params=True, display_macs=True)

    optimizer = oto.geta(
        variant=variant,
        lr=lr,
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
        min_bit_wt=min_bit_wt,
        max_bit_wt=max_bit_wt,
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
    if config.num_gpus > 1:
        logger.info(f"Using {config.num_gpus} GPUs for training")
        model = nn.DataParallel(model)

    best_epoch = 0
    best_acc1 = 0.0
    loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        ):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                if label_smooth and not mix_up:
                    targets = one_hot(targets, num_classes=10, smoothing_eps=0.1)
                if not label_smooth and mix_up:
                    targets = one_hot(targets, num_classes=10)
                    inputs, targets = mixup_func(inputs, targets)
                if mix_up and label_smooth:
                    targets = one_hot(targets, num_classes=10, smoothing_eps=0.1)
                    inputs, targets = mixup_func(inputs, targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            running_loss += loss.item()

            # Parameter information
            # with open("./log_new/param_info.txt", "a") as f1:
            #     f1.write(f"Epoch: {epoch}, Batch:{batch_idx}\n")
            # opt_metrics_epoch = optimizer.compute_metrics()

        lr_scheduler.step()

        opt_metrics = optimizer.compute_metrics()
        running_loss_avg = running_loss / len(train_loader)

        accuracy1, accuracy5 = check_accuracy(
            model.module if config.num_gpus > 1 else model, test_loader, two_input=False
        )
        avg_wt_bit = oto.compute_average_bit_width()
        logger.info(
            f"GPU{config.local_rank}, Epoch: {epoch}, loss: {running_loss_avg:5.3f}, norm_all: {opt_metrics.norm_params:5.2f}, grp_sparsity: {opt_metrics.group_sparsity:5.2f}, acc1: {accuracy1:5.2f}%, acc5: {accuracy5:5.2f}%, norm_import: {opt_metrics.norm_important_groups:5.2f}, norm_redund: {opt_metrics.norm_redundant_groups:5.2f}, num_grp_import: {opt_metrics.num_important_groups:5.2f}, num_grp_redund: {opt_metrics.num_redundant_groups:5.2f}, avg_wt_bit_width: {avg_wt_bit:5.2f}"
        )

        if accuracy1 > best_acc1 and config.local_rank == 0:
            best_acc1 = accuracy1
            best_epoch = epoch
            torch.save(model, os.path.join(output_dir, "best_acc1.pt"))

        loss_list.append(running_loss_avg)

    logger.info(f"Best epoch: {best_epoch}. Best acc1: {best_acc1}%")
    logger.info("Training completed. Constructing subnet...")

    # Construct the subnet and get the compressed model
    if config.local_rank == 0:
        oto.construct_subnet(out_dir=os.path.join(output_dir, "subnet"))
        compressed_model = torch.load(oto.compressed_model_path)
        oto_compressed = OTO(compressed_model.to(device), dummy_input.to(device))

        logger.info(f"Full MACs for Q{model_name}: {full_macs['total']} M MACs")
        logger.info(f"Full BOPs for Q{model_name}: {full_bops['total']} M BOPs")
        logger.info(f"Full num params for Q{model_name}: {full_num_params} M params")
        logger.info(
            f"Full weight size for Q{model_name}: {full_weight_size['total']} MB"
        )
        if "layer_info" in full_macs and "layer_info" in full_bops:
            logger.info("Layer-by-layer breakdown for full model:")
            logger.info(f"{'Layer':<30} {'Type':<15} {'MACs (M)':<15} {'BOPs (M)':<15}")
            logger.info("-" * 75)
            for mac_info, bop_info in zip(
                full_macs["layer_info"], full_bops["layer_info"]
            ):
                logger.info(
                    f"{mac_info['name']:<30} {mac_info['type']:<15} {mac_info['macs']:<15.2f} {bop_info['bops']:<15.2f}"
                )

        # Get compressed model MACs, BOPs, and number of parameters
        compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
        compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)
        compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)

        logger.info(
            f"Compressed MACs for Q{model_name}: {compressed_macs['total']} M MACs"
        )
        logger.info(
            f"Compressed BOPs for Q{model_name}: {compressed_bops['total']} M BOPs"
        )
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
        default="imagenet",
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--lr_quant", type=float, default=1e-3, help="Initial learning rate"
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
        "--min_bit_wt", type=int, default=4, help="bit width range minimum"
    )
    parser.add_argument(
        "--max_bit_wt", type=int, default=16, help="bit width range maximum"
    )
    parser.add_argument(
        "--min_bit_act", type=int, default=4, help="bit width range minimum"
    )
    parser.add_argument(
        "--max_bit_act", type=int, default=16, help="bit width range maximum"
    )
    parser.add_argument("--mix_up", type=int, default=0, help="mixup yes(1) or no(0)")
    parser.add_argument(
        "--label_smooth", type=int, default=0, help="label smoothing yes(1) or no(0)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--ddp", type=bool, default=False, help="enable ddp")
    parser.add_argument(
        "--train_dir", type=str, default="", help="Training data directory"
    )
    parser.add_argument(
        "--test_dir", type=str, default="", help="Testing data directory"
    )
    # Parse arguments
    config = add_common_args(parser).parse_args()

    return config


if __name__ == "__main__":
    main(get_config())
