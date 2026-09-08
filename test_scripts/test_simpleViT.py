# HESSO on resnet56 Cifar10 dataset
import argparse

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from only_train_once import OTO
from sanity_check.backends.simple_vit import simpleViT_cifar10


def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Sparsity",
    )

    # Parse arguments
    config = parser.parse_args()

    return config


def main(config):
    model = simpleViT_cifar10()
    dummy_input = torch.rand(1, 3, 32, 32)
    oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())
    oto.mark_unprunable_by_param_names(["to_patch_embedding.2.weight"])

    # A ResNet_zig.gv.pdf will be generated to display the depandancy graph.
    oto.visualize(view=False, out_dir="../cache")

    trainset = CIFAR10(
        root="cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    testset = CIFAR10(
        root="cifar10",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4
    )

    optimizer = oto.hesso(
        variant="sgd",
        lr=0.1,
        weight_decay=1e-4,
        target_group_sparsity=config.sparsity,
        start_pruning_step=200 * len(trainloader),
        pruning_periods=10,
        pruning_steps=10 * len(trainloader),
    )

    from utils.utils import check_accuracy

    max_epoch = 200
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # Every 50 epochs, decay lr by 10.0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

    for epoch in range(max_epoch):
        f_avg_val = 0.0
        model.train()
        lr_scheduler.step()
        for X, y in trainloader:
            X = X.cuda()
            y = y.cuda()
            y_pred = model.forward(X)
            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f
            optimizer.step()
        opt_metrics = optimizer.compute_metrics()
        # group_sparsity, param_norm, _ = optimizer.compute_group_sparsity_param_norm()
        # norm_important, norm_redundant, num_grps_important, num_grps_redundant = optimizer.compute_norm_groups()
        accuracy1, accuracy5 = check_accuracy(model, testloader)
        f_avg_val = f_avg_val.cpu().item() / len(trainloader)

        print(
            f"Ep: {epoch}, loss: {f_avg_val:.2f}, norm_all:{opt_metrics.norm_params:.2f}, grp_sparsity: {opt_metrics.group_sparsity:.2f}, acc1: {accuracy1:.4f}, norm_import: {opt_metrics.norm_important_groups:.2f}, norm_redund: {opt_metrics.norm_redundant_groups:.2f}, num_grp_import: {opt_metrics.num_important_groups}, num_grp_redund: {opt_metrics.num_redundant_groups}"
        )

    # save the .pt file
    torch.save(model, "simpleViT_checkpoint.pt")


if __name__ == "__main__":
    main(get_config())
