"""
Model description on Huggingface
https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k

Hotfix: Need to comment out line 186 of the pruning_dependency.py file
"""

import os
import unittest

import torch
from backends.vision_transformer.Swin import swin_tiny_patch4_window7_224
from torch import nn

from only_train_once import OTO

OUT_DIR = "./cache"


class TestSwin(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):
        model = swin_tiny_patch4_window7_224(pretrained=False, num_classes=1000)
        model.head = nn.Linear(model.head.in_features, 10)

        oto = OTO(model, dummy_input=dummy_input)

        unprunable_list = ["patch_embed.proj.weight", "pos_embed"]
        for name, param in model.named_parameters():
            if "attn.qkv." in name:
                unprunable_list.append(name)

        # print(unprunable_list)
        # exit()

        oto.mark_unprunable_by_param_names(unprunable_list)

        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        oto.random_set_zero_groups(target_group_sparsity=0.8)

        oto.construct_subnet(
            export_huggingface_format=False,
            export_float16=False,
            full_group_sparse_model_dir=OUT_DIR,
            compressed_model_dir=OUT_DIR,
        )

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference " + str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )

        # # For test FLOP and param reductions.
        # oto_compressed = OTO(compressed_model, dummy_input)
        # compressed_flops = oto_compressed.compute_flops(in_million=True)['total']
        # compressed_num_params = oto_compressed.compute_num_params(in_million=True)

        # print("FLOP  reduction (%)    : ", 1.0 - compressed_flops / full_flops)
        # print("Param reduction (%)    : ", 1.0 - compressed_num_params / full_num_params)
