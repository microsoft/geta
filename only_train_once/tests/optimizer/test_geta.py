import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from only_train_once import OTO
from only_train_once.optimizer.utils import (
    load_checkpoint,
    save_checkpoint,
    scan_checkpoint,
)
from only_train_once.quantization.quant_model import model_to_quantize_model
from sanity_check.backends.vgg7 import vgg7_bn


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _ = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def check_accuracy(model, testloader, two_input=False):
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            if two_input:
                y_pred = model.forward(X, X)
            else:
                y_pred = model.forward(X)
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))

            correct1 += prec1.item()
            correct5 += prec5.item()

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def transforms_train():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture
def transforms_test():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture
def train_loader(transforms_train):
    trainset = CIFAR10(
        root="cifar10", train=True, download=True, transform=transforms_train
    )
    return DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)


@pytest.fixture
def test_loader(transforms_test):
    testset = CIFAR10(
        root="cifar10", train=False, download=True, transform=transforms_test
    )
    return DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)


@pytest.fixture
def model(device):
    base_model = vgg7_bn()
    model = model_to_quantize_model(base_model)
    return model.to(device)


@pytest.fixture
def optimizer(model, device, train_loader):
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model, dummy_input=dummy_input)
    return oto.geta(
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        projection_steps=10,
        start_pruning_step=10,
        pruning_steps=10,
    )


@pytest.fixture
def checkpoint_dir(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def test_model_setup(model, device):
    assert isinstance(model, torch.nn.Module)
    # Check if model parameters are on any CUDA device when CUDA is available
    if device.type == "cuda":
        assert next(model.parameters()).device.type == device.type
    else:
        assert next(model.parameters()).device == device


def test_optimizer_setup(optimizer):
    print(optimizer.bit_reduction)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert optimizer.target_group_sparsity == 0.5
    assert optimizer.max_bit_wt == 16
    assert optimizer.min_bit_wt == 2
    assert optimizer.max_bit_act == 16
    assert optimizer.min_bit_act == 2
    assert optimizer.bit_reduction == 2
    assert optimizer.start_projection_step == 0
    assert optimizer.start_pruning_step == 10
    assert optimizer.projection_periods == 1
    assert optimizer.pruning_periods == 1
    assert optimizer.projection_period_duration == 10
    assert optimizer.pruning_period_duration == 10
    assert optimizer.projection_steps == 10
    assert optimizer.pruning_steps == 10


def test_save_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint = optimizer.create_checkpoint(model, epoch=0, loss=0.0)
    checkpoint_path = checkpoint_dir / f"cifar10_vgg7_bn_{optimizer.num_steps}.pt"
    save_checkpoint(checkpoint_path, checkpoint)
    assert checkpoint_path.exists()


def test_load_checkpoint(model, optimizer, checkpoint_dir, device):
    # First save a checkpoint
    initial_state = model.state_dict()
    optimizer_state = optimizer.state_dict()  # This includes param_groups
    checkpoint = {
        "model_state_dict": initial_state,
        "optimizer_state_dict": optimizer_state,
        "epoch": 0,
        "loss": 0.0,
    }
    checkpoint_path = checkpoint_dir / f"cifar10_vgg7_bn_{optimizer.num_steps}.pt"
    save_checkpoint(checkpoint_path, checkpoint)

    # Then try to load it
    loaded_checkpoint = scan_checkpoint(checkpoint_dir, "cifar10_vgg7_bn_")
    assert loaded_checkpoint is not None

    state_dict = load_checkpoint(loaded_checkpoint, device)
    assert state_dict is not None
    assert "model_state_dict" in state_dict
    assert "optimizer_state_dict" in state_dict
    assert state_dict["optimizer_state_dict"]["param_groups"] is not None
    assert state_dict["optimizer_state_dict"]["num_steps"] == 0

    # Verify model state
    for key in initial_state.keys():
        assert torch.equal(initial_state[key], state_dict["model_state_dict"][key])

    # Verify optimizer state
    loaded_opt_state = state_dict["optimizer_state_dict"]
    assert loaded_opt_state["pruning_periods"] == optimizer.pruning_periods
    assert loaded_opt_state["start_projection_step"] == optimizer.start_projection_step
    assert loaded_opt_state["projection_periods"] == optimizer.projection_periods


@pytest.mark.slow
def test_training_step(model, optimizer, train_loader, device):
    model.train()

    # Get a single batch
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)

    # Forward pass
    y_pred = model(X)
    loss = torch.nn.CrossEntropyLoss()(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0


# @pytest.mark.slow
def test_training_loop(
    model, optimizer, train_loader, test_loader, device, checkpoint_dir
):
    cp = None
    if checkpoint_dir is not None:
        print(f"Checking storage path: {checkpoint_dir}")
        try:
            cp = scan_checkpoint(checkpoint_dir, "carn_")
        except Exception as e:
            print(f"Error scanning storage checkpoint path: {e}")

    max_steps = 2  # Small number for testing
    save_freq = 1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    step = 0
    model.train()

    if cp is None:
        state_dict = None
        last_epoch = -1
        last_loss = None
        print("No checkpoint found. Starting from scratch.")
    else:
        print(f"Loading checkpoint: {cp}")
        state_dict = load_checkpoint(cp, device)
        model.load_state_dict(state_dict["model_state_dict"])
        # optimizer.load_state_dict() correctly loads expected state
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        step = optimizer.num_steps
        last_epoch = state_dict["epoch"]
        last_loss = state_dict["loss"]  # Store the loss value from the checkpoint
        print(
            f"Successfully resumed from epoch {last_epoch}, step {step}, loss {last_loss}"
        )

    for X, y in train_loader:
        if step >= max_steps:
            break

        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = torch.nn.CrossEntropyLoss()(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % save_freq == 0:
            # Save checkpoint
            checkpoint = optimizer.create_checkpoint(model, epoch=0, loss=loss.item())
            checkpoint_path = (
                checkpoint_dir / f"cifar10_vgg7_bn_{optimizer.num_steps}.pt"
            )
            save_checkpoint(checkpoint_path, checkpoint)

            # Compute metrics
            optimizer.compute_metrics()
            accuracy1, accuracy5 = check_accuracy(model, test_loader, device)
            assert 0 <= accuracy1 <= 100
            assert 0 <= accuracy5 <= 100

        step += 1
        if step % len(train_loader) == 0:
            lr_scheduler.step()

    assert step == max_steps
