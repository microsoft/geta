import os
import unittest

import torch
from mamba_ssm import Mamba

from only_train_once import OTO

OUT_DIR = "./cache"


class TestMamba(unittest.TestCase):
    def test_sanity(self):
        batch, length, dim = 2, 64, 16
        dummy_input = torch.randn(batch, length, dim).to("cuda")
        model = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")
        y = model(dummy_input)
        assert y.shape == dummy_input.shape
        for name, param in model.named_parameters():
            print(name, param.shape)
        # torch.onnx.export(
        #     model,
        #     dummy_input,
        #     os.path.join(OUT_DIR, 'mamba.onnx')
        # )
        exit()
        oto = OTO(model, dummy_input)
        oto.visualize(view=False, out_dir=OUT_DIR)
        # For test FLOP and param reductions.
        full_flops = oto.compute_flops(in_million=True)["total"]
        full_num_params = oto.compute_num_params(in_million=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
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

        # For test FLOP and param reductions.
        oto_compressed = OTO(compressed_model, dummy_input)
        compressed_flops = oto_compressed.compute_flops(in_million=True)["total"]
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)

        print("FLOP  reduction (%)    : ", 1.0 - compressed_flops / full_flops)
        print(
            "Param reduction (%)    : ", 1.0 - compressed_num_params / full_num_params
        )
