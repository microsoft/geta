import datetime
import glob
import os
import random
import subprocess
import zipfile

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends import CarnNet


def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    hsize = size * scale
    hx, hy = x * scale, y * scale
    crop_lr = lr[y : y + size, x : x + size].copy()
    crop_hr = hr[hy : hy + hsize, hx : hx + hsize].copy()
    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1, im2 = np.flipud(im1), np.flipud(im2)
    if random.random() < 0.5:
        im1, im2 = np.fliplr(im1), np.fliplr(im2)
    angle = random.choice([0, 1, 2, 3])
    im1, im2 = np.rot90(im1, angle), np.rot90(im2, angle)
    return im1.copy(), im2.copy()


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        return (im.astype(np.float64) - min_val) / (max_val - min_val)

    im1, im2 = im2double(im1), im2double(im2)
    return peak_signal_noise_ratio(im1, im2, data_range=1)


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super().__init__()
        self.size = size
        h5f = h5py.File(path, "r")
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.scale = [scale] if scale != 0 else [2, 3, 4]
        self.lr = [[v[:] for v in h5f[f"X{s}"].values()] for s in self.scale]
        h5f.close()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        size = self.size
        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        item = [
            random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)
        ]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]
        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale):
        super().__init__()
        self.name = dirname.split("/")[-1]
        print(f"Test set name is {self.name}")
        self.scale = scale
        if "DIV" in self.name:
            self.hr = glob.glob(os.path.join(f"{dirname}_HR", "*.png"))
            self.lr = glob.glob(
                os.path.join(f"{dirname}_LR_bicubic", f"X{scale}/*.png")
            )
        else:
            all_files = glob.glob(os.path.join(dirname, f"x{scale}/*.png"))
            self.hr = [name for name in all_files if "HR" in name]
            self.lr = [name for name in all_files if "LR" in name]
        self.hr.sort()
        self.lr.sort()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index]).convert("RGB")
        lr = Image.open(self.lr[index]).convert("RGB")
        filename = self.hr[index].split("/")[-1]
        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)


def download_and_preprocess_dataset():
    dataset_dir = "./outputs/carn_sr/"
    os.makedirs(dataset_dir, exist_ok=True)
    # Download dataset
    urls = [
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
    ]

    for url in urls:
        filename = url.split("/")[-1]
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-P", dataset_dir, url], check=True)

    for zip_file in glob.glob(os.path.join(dataset_dir, "*.zip")):
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            print(f"Extracting {zip_file}...")
            zip_ref.extractall(dataset_dir)

    dataset_type = "train"
    h5_file = os.path.join(dataset_dir, f"DIV2K_{dataset_type}.h5")

    if not os.path.exists(h5_file):
        with h5py.File(h5_file, "w") as f:
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            for subdir in ["HR", "X2", "X3", "X4"]:
                if subdir in ["HR"]:
                    im_paths = glob.glob(
                        os.path.join(dataset_dir, f"DIV2K_{dataset_type}_HR", "*.png")
                    )
                else:
                    im_paths = glob.glob(
                        os.path.join(
                            dataset_dir,
                            f"DIV2K_{dataset_type}_LR_bicubic",
                            subdir,
                            "*.png",
                        )
                    )
                im_paths.sort()
                grp = f.create_group(subdir)

                for i, path in tqdm(enumerate(im_paths), desc=f"Processing {subdir}"):
                    im = cv2.imread(path)
                    grp.create_dataset(str(i), data=im)

    print("Dataset downloaded and preprocessed successfully.")


def evaluate(model, test_data_dir, scale=2):
    shave = 20
    mean_psnr = 0
    model.eval()
    print(f"Evaluating on {test_data_dir}...")
    test_data = TestDataset(test_data_dir, scale=scale)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=False)

    for step, inputs in enumerate(test_loader):
        hr, lr, name = inputs[0].squeeze(0), inputs[1].squeeze(0), inputs[2][0]
        h, w = lr.size()[1:]
        h_half, w_half = int(h / 2), int(w / 2)
        h_chop, w_chop = h_half + shave, w_half + shave

        lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop : w])
        lr_patch[2].copy_(lr[:, h - h_chop : h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h - h_chop : h, w - w_chop : w])
        lr_patch = lr_patch.cuda()

        sr = model(lr_patch, scale).data

        h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
        w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

        result = torch.FloatTensor(3, h, w).cuda()
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(
            sr[1, :, 0:h_half, w_chop - w + w_half : w_chop]
        )
        result[:, h_half:h, 0:w_half].copy_(
            sr[2, :, h_chop - h + h_half : h_chop, 0:w_half]
        )
        result[:, h_half:h, w_half:w].copy_(
            sr[3, :, h_chop - h + h_half : h_chop, w_chop - w + w_half : w_chop]
        )
        sr = result

        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        bnd = scale
        im1 = hr[bnd:-bnd, bnd:-bnd]
        im2 = sr[bnd:-bnd, bnd:-bnd]
        im1_y = rgb2ycbcr(im1)[..., 0]
        im2_y = rgb2ycbcr(im2)[..., 0]

        mean_psnr += psnr(im1_y, im2_y) / len(test_data)
    return mean_psnr


def train_carn():
    log_dir = "./outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    scale = 2
    model = CarnNet(scale=scale, multi_scale=False, group=1)
    model = model_to_quantize_model(model)
    print("Converted float model to quantized model.")

    dummy_input = torch.rand(1, 3, 224, 224)
    oto = OTO(model.cuda(), (dummy_input.cuda(), scale))
    oto.mark_unprunable_by_param_names(["exit.weight"])

    download_and_preprocess_dataset()
    train_data = TrainDataset(
        "./outputs/carn_sr/DIV2K_train.h5",
        scale=2,
        size=64,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    optimizer = oto.geta(
        variant="adam",
        lr=1e-4,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        # start_pruning_step=10 * len(train_loader), #100, 1000
        start_pruning_step=1000,
        pruning_periods=10,
        # pruning_steps=10 * len(train_loader),
        pruning_steps=1000,
    )
    max_step = 600000
    lr_decay_step = 400000
    print_interval = 100
    loss_fn = torch.nn.L1Loss()
    f_avg_val = 0.0
    learning_rate = optimizer.get_learning_rate()
    print("Starting training...")

    step = 0
    while True:
        for inputs in train_loader:
            model.train()
            hr, lr = inputs[-1][0].cuda(), inputs[-1][1].cuda()
            optimizer.zero_grad()
            sr = model(lr, scale)
            loss = loss_fn(sr, hr)
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Error during backward pass: {e!s}")
                with open(log_file, "a") as f:
                    f.write(f"Error during backward pass: {e!s}\n")
                print("Continuing to next step...")
                continue
            optimizer.grad_clipping()
            optimizer.step()
            learning_rate = learning_rate * (0.5 ** (step // lr_decay_step))
            optimizer.set_learning_rate(learning_rate)
            f_avg_val += loss.item()
            if step % print_interval == 0:
                print(f"Step: {step}, loss: {f_avg_val / print_interval:.4f}")
                metrics = optimizer.compute_metrics()
                print(metrics)
                f_avg_val = 0.0
                psnr_B100 = evaluate(model, "./outputs/SR_benchmark/B100", scale=2)
                psnr_Set14 = evaluate(model, "./outputs/SR_benchmark/Set14", scale=2)
                psnr_Urban100 = evaluate(
                    model, "./outputs/SR_benchmark/Urban100", scale=2
                )
                print(
                    f"Val PSNR: B100: {psnr_B100:.4f}, Set14: {psnr_Set14:.4f}, Urban100: {psnr_Urban100:.4f}."
                )
            step += 1
        if step > max_step:
            break

    return oto, model


if __name__ == "__main__":
    oto, trained_model = train_carn()
    full_macs = oto.compute_macs(in_million=True, layerwise=True)
    full_bops = oto.compute_bops(in_million=True, layerwise=True)
    full_num_params = oto.compute_num_params(in_million=True)
    # Construct a compressed model and compute MACs, BOPs, and number of parameters
    oto.construct_subnet(out_dir="./outputs")
    compressed_model = torch.load(oto.compressed_model_path)
    oto_compressed = OTO(compressed_model, (torch.rand(1, 3, 224, 224).cuda(), 2))
    print(f"Full MACs for Full QCARN: {full_macs['total']} M MACs")
    print(f"Full BOPs for Full QCARN: {full_bops['total']} M BOPs")
    print(f"Full num params for Full QCARN: {full_num_params} M params")
    oto.print_layer_breakdown(full_macs["layer_info"], full_bops["layer_info"])
    compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
    compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
    compressed_num_params = oto_compressed.compute_num_params(in_million=True)
    print(f"Compressed MACs for Compressed QCARN: {compressed_macs['total']} M MACs")
    print(f"Compressed BOPs for Compressed QCARN: {compressed_bops['total']} M BOPs")
    print(
        f"Compressed num params for Compressed QCARN: {compressed_num_params} M params"
    )
    oto_compressed.print_layer_breakdown(
        compressed_macs["layer_info"], compressed_bops["layer_info"]
    )
    print(f"MAC reduction (%): {1.0 - compressed_macs['total'] / full_macs['total']}")
    print(f"BOP reduction (%): {1.0 - compressed_bops['total'] / full_bops['total']}")
    print(f"Param reduction (%): {1.0 - compressed_num_params / full_num_params}")
    print(f"MAC ratio: {full_macs['total'] / compressed_macs['total']}")
    print(f"BOP compression ratio: {full_bops['total'] / compressed_bops['total']}")
    full_model_size = os.path.getsize(oto.full_group_sparse_model_path) / (1024**3)
    compressed_model_size = os.path.getsize(oto.compressed_model_path) / (1024**3)
    print(f"Size of full model: {full_model_size:.4f} GB")
    print(f"Size of compressed model: {compressed_model_size:.4f} GB")
