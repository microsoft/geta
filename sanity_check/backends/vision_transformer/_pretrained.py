from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["DefaultCfg", "PretrainedCfg", "filter_pretrained_cfg"]


@dataclass
class PretrainedCfg:
    """ """

    # weight source locations
    url: str | tuple[str, str] | None = None  # remote URL
    file: str | None = None  # local / shared filesystem path
    state_dict: dict[str, Any] | None = None  # in-memory state dict
    hf_hub_id: str | None = None  # Hugging Face Hub model id ('organization/model')
    hf_hub_filename: str | None = None  # Hugging Face Hub filename (overrides default)

    source: str | None = (
        None  # source of cfg / weight location used (url, file, hf-hub)
    )
    architecture: str | None = None  # architecture variant can be set when not implicit
    tag: str | None = None  # pretrained tag of source
    custom_load: bool = (
        False  # use custom model specific model.load_pretrained() (ie for npz files)
    )

    # input / data config
    input_size: tuple[int, int, int] = (3, 224, 224)
    test_input_size: tuple[int, int, int] | None = None
    min_input_size: tuple[int, int, int] | None = None
    fixed_input_size: bool = False
    interpolation: str = "bicubic"
    crop_pct: float = 0.875
    test_crop_pct: float | None = None
    crop_mode: str = "center"
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    # head / classifier config and meta-data
    num_classes: int = 1000
    label_offset: int | None = None
    label_names: tuple[str] | None = None
    label_descriptions: dict[str, str] | None = None

    # model attributes that vary with above or required for pretrained adaptation
    pool_size: tuple[int, ...] | None = None
    test_pool_size: tuple[int, ...] | None = None
    first_conv: str | None = None
    classifier: str | None = None

    license: str | None = None
    description: str | None = None
    origin_url: str | None = None
    paper_name: str | None = None
    paper_ids: str | tuple[str] | None = None
    notes: tuple[str] | None = None

    @property
    def has_weights(self):
        return self.url or self.file or self.hf_hub_id

    def to_dict(self, remove_source=False, remove_null=True):
        return filter_pretrained_cfg(
            asdict(self), remove_source=remove_source, remove_null=remove_null
        )


def filter_pretrained_cfg(cfg, remove_source=False, remove_null=True):
    filtered_cfg = {}
    keep_null = {
        "pool_size",
        "first_conv",
        "classifier",
    }  # always keep these keys, even if none
    for k, v in cfg.items():
        if remove_source and k in {
            "url",
            "file",
            "hf_hub_id",
            "hf_hub_filename",
            "source",
        }:
            continue
        if remove_null and v is None and k not in keep_null:
            continue
        filtered_cfg[k] = v
    return filtered_cfg


@dataclass
class DefaultCfg:
    tags: deque[str] = field(
        default_factory=deque
    )  # priority queue of tags (first is default)
    cfgs: dict[str, PretrainedCfg] = field(
        default_factory=dict
    )  # pretrained cfgs by tag
    is_pretrained: bool = (
        False  # at least one of the configs has a pretrained source set
    )

    @property
    def default(self):
        return self.cfgs[self.tags[0]]

    @property
    def default_with_tag(self):
        tag = self.tags[0]
        return tag, self.cfgs[tag]
