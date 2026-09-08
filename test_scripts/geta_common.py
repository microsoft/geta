import logging
import os
import sys

from torch.utils.data import IterableDataset


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_output_dir(explicit=None, label="run"):
    if explicit:
        out = explicit
    else:
        out = os.environ.get(
            "GETA_OUTPUT_DIR", os.path.join(repo_root(), "outputs", label)
        )
    os.makedirs(out, exist_ok=True)
    return out


def resolve_data_dir(explicit=None):
    if explicit:
        return explicit
    return os.environ.get("GETA_DATA_DIR", os.path.join(repo_root(), "data"))


def add_common_args(parser):
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    return parser


def create_exp_dir(config, outputs="outputs", exp_name="exp"):
    base = getattr(config, "output_dir", None) or os.environ.get("GETA_OUTPUT_DIR")
    if not base:
        base = os.path.join(repo_root(), outputs, exp_name)
    os.makedirs(base, exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(base, f"{exp_name}.log"))
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(sh)
    return logger


class StreamingDataset(IterableDataset):
    def __init__(
        self, hf_dataset, preprocess_func, length, max_samples_per_epoch=100000
    ):
        self.hf_dataset = hf_dataset
        self.preprocess_func = preprocess_func
        self.length = length
        self.max_samples_per_epoch = max_samples_per_epoch

    def __iter__(self):
        count = 0
        for example in self.hf_dataset:
            if self.max_samples_per_epoch and count >= self.max_samples_per_epoch:
                break
            yield self.preprocess_func(example)
            count += 1

    def __len__(self):
        return self.length
