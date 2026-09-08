import os
import unittest

import torch
from backends import MambaLM, MambaLMConfig

from only_train_once import OTO

OUT_DIR = "./cache"


class TestMamba(unittest.TestCase):
    def test_sanity(self):
        batch, length, dim = 2, 64, 16
        dummy_input = torch.randn(batch, length, dim).to("cuda")
        config = MambaLMConfig(d_model=128, n_layers=2)
        # config.n_layers = 4
        model = MambaLM(config)
        # model = mamba_from_pretrained('state-spaces/mamba-370m')
        # model.eval()
        # torch.onnx.export(model,
        #           (torch.zeros(1, dtype=torch.int64), *model.init_caches()),
        #           os.path.join(OUT_DIR, 'mamba-1-layers.onnx'),
        #           input_names=['input', 'hs', 'inputs'],
        #           output_names=['output', 'hs', 'inputs'],
        #           opset_version=17,
        #           )

        dummy_input = (torch.zeros(1, dtype=torch.int64), *model.init_caches())
        oto = OTO(model, dummy_input, strict_out_nodes=False)
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        return
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
