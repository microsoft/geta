#!/bin/bash
# Local (non-Slurm) sequential runner for the GETA CIFAR-10 baselines.
# Paper hyper-parameters are from Table 7 of geta.pdf.
# For cluster runs use ../run.sh with sbatch instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${GETA_DATA_DIR:-${REPO_ROOT}/data}"
OUT_ROOT="${GETA_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
mkdir -p "${DATA_DIR}" "${OUT_ROOT}"

run() {
    local label="$1"; shift
    echo "=== [${label}] $* ==="
    pixi run --environment geta python "$@" \
        --output_dir "${OUT_ROOT}/${label}" \
        --data_dir "${DATA_DIR}"
}

# Paper Table 2: ResNet20 on CIFAR10 (weight-only quant, SGD lr=0.1, 350 epochs)
run resnet20-cifar10 test_scripts/Qtest_resnet56_ablation.py \
    --model_name resnet20 --dataset cifar10 --batch_size 64 --epochs 350 \
    --lr 0.1 --lr_quant 1e-4 --weight_decay 1e-4 --sparsity 0.35 \
    --projection_start_step 0 --projection_periods 7 --projection_steps 35 \
    --pruning_start_step 35 --pruning_periods 5 --pruning_steps 30 \
    --variant sgd --bit_reduction 2 --min_bit_wt 4 --max_bit_wt 16 --seed 0

# Paper Table 4: VGG7 on CIFAR10 (weight+activation quant, Adam lr=1e-3, 200 epochs)
run vgg7-cifar10 test_scripts/Qtest_JPQ.py \
    --model_name vgg7bn --dataset cifar10 --batch_size 64 --epochs 200 \
    --learning_rate 1e-3 --weight_decay 1e-4 --sparsity_level 0.7 \
    --projection_start_step 0 --projection_periods 5 --projection_steps 20 \
    --pruning_start_step 20 --pruning_periods 10 --pruning_steps 30 \
    --variant adam --bit_reduction 2 --min_bit 4 --max_bit 16 --seed 0

# Paper Fig 4 context: ResNet56 on CIFAR10 (200 epochs)
run resnet56-cifar10 test_scripts/Qtest_resnet56_ablation.py \
    --model_name resnet56 --dataset cifar10 --batch_size 64 --epochs 200 \
    --lr 0.1 --lr_quant 1e-4 --weight_decay 1e-4 --sparsity 0.5 \
    --projection_start_step 0 --projection_periods 5 --projection_steps 20 \
    --pruning_start_step 20 --pruning_periods 10 --pruning_steps 30 \
    --variant sgd --bit_reduction 2 --min_bit_wt 4 --max_bit_wt 16 --seed 0

# Paper Table 6: SimpleViT on CIFAR10 (100 epochs)
run simplevit-cifar10 test_scripts/Qtest_simpleViT.py \
    --model_name simpleViT --dataset cifar10 --batch_size 64 --epochs 100 \
    --learning_rate 1e-3 --weight_decay 1e-4 --sparsity_level 0.3 \
    --projection_start_step 5 --projection_periods 5 --projection_steps 5 \
    --pruning_start_step 10 --pruning_periods 10 --pruning_steps 20 \
    --variant adam --bit_reduction 2 --min_bit 4 --max_bit 16 \
    --init_bit 16 --seed 0

echo "All baseline runs finished."
