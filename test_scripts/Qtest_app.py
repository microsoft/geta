import logging
import math
import os

import numpy as np
import torch
import typer
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from transformers import AutoImageProcessor

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
from sanity_check.backends.resnet_cifar10 import resnet18_cifar10
from sanity_check.backends.vgg7 import vgg7_bn
from test_scripts.geta_common import (
    StreamingDataset,
    resolve_data_dir,
    resolve_output_dir,
)
from utils.utils import check_accuracy

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer()


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
    # Compute the bit width
    for key in param_dict.keys():
        d = param_dict[key]["d_quant"]
        t = param_dict[key]["t_quant"]
        qmax = abs(param_dict[key]["q_m"])
        bit_width = math.log2(math.exp(t * math.log(qmax)) / abs(d) + 1) + 1
        bit_dict[key] = bit_width
    return bit_dict


def compute_bop_compression_ratio(
    original,
    compressed,
    bitwidths,
    input_size=(1, 3, 32, 32),
    original_weight_bitwidth=32,
    activation_bitwidth=32,  # Fixed activation bitwidth for both models
    verbose=True,
):
    total_original_bop = 0
    total_compressed_bop = 0
    total_original_mac = 0
    total_compressed_mac = 0
    current_input_size = input_size
    prev_original_channels = input_size[1]
    prev_compressed_channels = input_size[1]
    prev_pl = 0  # Initial pruning ratio
    layer_idx = 0
    for (name, original_layer), (_, compressed_layer) in zip(
        original.named_modules(), compressed.named_modules()
    ):
        if isinstance(original_layer, (nn.Conv2d, nn.Linear)) and name in bitwidths:
            # Compute pruning ratios
            original_channels = (
                original_layer.out_channels
                if isinstance(original_layer, nn.Conv2d)
                else original_layer.out_features
            )
            compressed_channels = (
                compressed_layer.out_channels
                if isinstance(compressed_layer, nn.Conv2d)
                else compressed_layer.out_features
            )
            pl = 1 - (compressed_channels / original_channels)
            Pl = 1 - (1 - prev_pl) * (1 - pl)  # Layerwise pruning ratio

            # Compute dimensions
            if isinstance(original_layer, nn.Conv2d):
                mw_l, mh_l = current_input_size[2], current_input_size[3]
                kw, kh = original_layer.kernel_size
            else:  # Linear layer
                mw_l, mh_l = 1, 1
                kw, kh = 1, 1

            # Compute MAC counts
            mac_original = (
                (1 - prev_pl)
                * prev_original_channels
                * (1 - pl)
                * original_channels
                * mw_l
                * mh_l
                * kw
                * kh
            )
            mac_compressed = (
                (1 - prev_pl)
                * prev_compressed_channels
                * (1 - pl)
                * compressed_channels
                * mw_l
                * mh_l
                * kw
                * kh
            )

            # Add MAC counts to totals
            total_original_mac += mac_original
            total_compressed_mac += mac_compressed

            # Compute BOP counts
            bw_l = round(bitwidths[name])
            bop_original = mac_original * original_weight_bitwidth * activation_bitwidth
            bop_compressed = mac_compressed * bw_l * activation_bitwidth

            total_original_bop += bop_original
            total_compressed_bop += bop_compressed

            if verbose:
                logger.info(f"Layer name: {name}, Layer index: {layer_idx}")
                logger.info(
                    f"Original channels: {original_channels}, Compressed channels: {compressed_channels}"
                )
                logger.info(
                    f"Pruning ratio (pl): {pl:.4f}, Layerwise pruning ratio (Pl): {Pl:.4f}"
                )
                logger.info(
                    f"MAC count - Original: {mac_original / 1e6:.4f} M, Compressed: {mac_compressed / 1e6:.4f} M"
                )
                logger.info(
                    f"BOP count - Original: {bop_original / 1e9:.4f} G, Compressed: {bop_compressed / 1e9:.4f} G"
                )
                logger.info(
                    f"Weight Bitwidth - Original: {original_weight_bitwidth}, Compressed: {bw_l}"
                )
                logger.info(f"Activation Bitwidth: {activation_bitwidth}")
                logger.info("--------------------")

            # Update for next layer
            prev_original_channels = original_channels
            prev_compressed_channels = compressed_channels
            prev_pl = pl
            if isinstance(original_layer, nn.Conv2d):
                current_input_size = (
                    current_input_size[0],
                    original_channels,
                    (
                        current_input_size[2]
                        + 2 * original_layer.padding[0]
                        - original_layer.kernel_size[0]
                    )
                    // original_layer.stride[0]
                    + 1,
                    (
                        current_input_size[3]
                        + 2 * original_layer.padding[1]
                        - original_layer.kernel_size[1]
                    )
                    // original_layer.stride[1]
                    + 1,
                )
            layer_idx += 1

    bop_compression_ratio = (
        total_original_bop / total_compressed_bop
        if total_compressed_bop > 0
        else float("inf")
    )
    mac_compression_ratio = (
        total_original_mac / total_compressed_mac
        if total_compressed_mac > 0
        else float("inf")
    )

    if verbose:
        logger.info(f"Total Original MAC: {total_original_mac / 1e9:.4f} GMACs")
        logger.info(f"Total Compressed MAC: {total_compressed_mac / 1e9:.4f} GMACs")
        logger.info(f"MAC Compression Ratio: {mac_compression_ratio:.4f}")
        logger.info(f"Total Original BOP: {total_original_bop / 1e9:.4f} GBOPs")
        logger.info(f"Total Compressed BOP: {total_compressed_bop / 1e9:.4f} GBOPs")
        logger.info(f"BOP Compression Ratio: {bop_compression_ratio:.4f}")

    return bop_compression_ratio, total_original_mac, total_compressed_mac


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
            root=os.path.join(resolve_data_dir(data_dir), "cifar10"),
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root=os.path.join(resolve_data_dir(data_dir), "cifar10"),
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
        # Load ImageNet dataset from Hugging Face
        train_dataset = load_dataset("imagenet-1k", split="train", streaming=True)
        val_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

        def preprocess_images(example):
            if isinstance(example["image"], Image.Image):
                image = example["image"]
            else:
                image = Image.fromarray(example["image"])
            image = image.convert("RGB")
            inputs = image_processor(images=np.array(image), return_tensors="pt")
            inputs["labels"] = torch.tensor(example["label"])
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": inputs["labels"].squeeze(),
            }

        train_length = 1281167
        val_length = 50000
        train_dataset = StreamingDataset(train_dataset, preprocess_images, train_length)
        val_dataset = StreamingDataset(val_dataset, preprocess_images, val_length)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers
        )
        input_size = (1, 3, 224, 224)
    else:
        raise ValueError("Unsupported dataset")

    return train_loader, test_loader, input_size


@app.command()
def main(
    model_name: str = typer.Option(
        "resnet18", help="Model architecture to use (resnet18)"
    ),
    dataset: str = typer.Option("cifar10", help="Dataset to use (cifar10 or imagenet)"),
    batch_size: int = typer.Option(256, help="Batch size for training"),
    num_workers: int = typer.Option(4, help="Number of workers for data loading"),
    epochs: int = typer.Option(20, help="Number of epochs to train"),
    learning_rate: float = typer.Option(0.1, help="Initial learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    sparsity_level: float = typer.Option(0.7, help="Target sparsity level"),
    pruningstep: int = typer.Option(1, help="Start pruning step"),
    pruning_periods: int = typer.Option(10, help="Number of pruning periods"),
    pruning_steps: int = typer.Option(10, help="Number of pruning steps"),
    lr_step: int = typer.Option(100, help="LR scheduler step size"),
    lr_gamma: float = typer.Option(0.1, help="LR scheduler gamma"),
    log_interval: int = typer.Option(100, help="How often to log training stats"),
    eval_interval: int = typer.Option(1, help="How often to evaluate the model"),
    variant: str = typer.Option("sgd", help="Optimizer variant"),
    seed: int = typer.Option(1, help="Random seed"),
    output_dir: str = typer.Option(
        None,
        help="Output directory for models (default: <repo>/outputs/<dataset>_<model_name>)",
    ),
    data_dir: str = typer.Option(
        None,
        help="Data directory (default: <repo>/data or $GETA_DATA_DIR)",
    ),
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    output_dir = resolve_output_dir(output_dir, f"{dataset}_{model_name}")

    train_loader, test_loader, input_size = get_data_loader(
        dataset, batch_size * num_gpus, num_workers, data_dir
    )
    num_classes = 10 if dataset == "cifar10" else 1000
    dummy_input = torch.rand(input_size).to(device)

    if model_name == "resnet18":
        model = resnet18_cifar10()
    elif model_name == "resnet20":
        model = resnet20_cifar10()
    elif model_name == "vgg7bn":
        model = vgg7_bn()
    else:
        raise ValueError("Unsupported model")

    model = model_to_quantize_model(model)
    oto = OTO(model.to(device), dummy_input=dummy_input)
    optimizer = oto.geta(
        variant=variant,
        lr=learning_rate,
        weight_decay=weight_decay,
        target_group_sparsity=sparsity_level,
        start_pruning_step=pruningstep * len(train_loader),
        pruning_periods=pruning_periods,
        pruning_steps=pruning_steps * len(train_loader),
    )
    # TODO: speed up distributed training
    # fabric = Fabric(accelerator='cuda')
    # fabric.launch()
    # model, optimizer = fabric.setup(model, optimizer)
    # train_loader = fabric.setup_dataloader(train_loader)
    # test_loader = fabric.setup_dataloader(test_loader)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_step, gamma=lr_gamma
    )
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs for training")
        model = nn.DataParallel(model)

    for epoch in range(epochs):
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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.grad_clipping()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss / log_interval:.4f}"
                )
                running_loss = 0.0

        lr_scheduler.step()

        if epoch % eval_interval == 0:
            accuracy1, accuracy5 = check_accuracy(
                model.module if num_gpus > 1 else model, test_loader, device
            )
            logger.info(
                f"Epoch: {epoch}, Top-1 Accuracy: {accuracy1:.2f}%, Top-5 Accuracy: {accuracy5:.2f}%"
            )

    logger.info("Training completed. Constructing subnet...")
    # Get full/original floating-point model MACs, BOPs, and number of parameters
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    # Construct the subnet and get the compressed model
    oto.construct_subnet(out_dir=output_dir)
    compressed_model = torch.load(oto.compressed_model_path)
    oto_compressed = OTO(compressed_model, dummy_input)

    logger.info(f"Full MACs for Q{model_name}: {full_macs['total']} M MACs")
    logger.info(f"Full BOPs for Q{model_name}: {full_bops['total']} M BOPs")
    logger.info(f"Full num params for Q{model_name}: {full_num_params} M params")

    oto.print_layer_breakdown(full_macs["layer_info"], full_bops["layer_info"])

    # Get compressed model MACs, BOPs, and number of parameters
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
    compressed_num_params = oto_compressed.compute_num_params(in_million=True)
    logger.info(f"Compressed MACs for Q{model_name}: {compressed_macs['total']} M MACs")
    logger.info(f"Compressed BOPs for Q{model_name}: {compressed_bops['total']} M BOPs")
    logger.info(
        f"Compressed num params for Q{model_name}: {compressed_num_params} M params"
    )

    oto_compressed.print_layer_breakdown(
        compressed_macs["layer_info"], compressed_bops["layer_info"]
    )

    logger.info(
        f"MAC reduction (%): {1.0 - compressed_macs['total'] / full_macs['total']}"
    )
    logger.info(
        f"BOP reduction (%): {1.0 - compressed_bops['total'] / full_bops['total']}"
    )
    logger.info(f"Param reduction (%): {1.0 - compressed_num_params / full_num_params}")
    logger.info(f"MAC ratio: {full_macs['total'] / compressed_macs['total']}")
    logger.info(
        f"BOP compresion ratio: {full_bops['total'] / compressed_bops['total']}"
    )

    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024**3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024**3)
    logger.info(f"Size of full/ model: {full_model_size:.4f} GB")
    logger.info(f"Size of compressed model: {compressed_model_size:.4f} GB")


if __name__ == "__main__":
    app()
