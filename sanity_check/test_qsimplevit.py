import unittest

import torch
from backends.simple_vit import simpleViT_cifar10

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model

OUT_DIR = "./cache"


class TestQSimpleViT(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 32, 32)):
        model = simpleViT_cifar10()
        q_model = model_to_quantize_model(model)
        oto = OTO(q_model, dummy_input)
        oto.mark_unprunable_by_param_names(["to_patch_embedding.2.weight"])
        # oto.visualize(view=False, out_dir=OUT_DIR, display_flops=True, display_params=True, display_macs=True)

        # For full model param reductions, MACs, and BOPs.
        full_num_params = oto.compute_num_params(in_million=True)
        full_macs = oto.compute_macs(in_million=True)["total"]
        full_bops = oto.compute_bops(in_million=True)["total"]
        full_weight_size = oto.compute_weight_size(in_million=True)["total"]
        print(full_macs)
        print(full_bops)
        print(full_num_params)
        print(full_weight_size)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        # For compressed model param reductions, MACs, and BOPs.
        oto_compressed = OTO(compressed_model, dummy_input)
        oto.visualize(
            view=False, out_dir=OUT_DIR, display_flops=True, display_params=True
        )
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)
        compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)[
            "total"
        ]
        compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)[
            "total"
        ]
        compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)[
            "total"
        ]

        print(
            "Param reduction (%)      : ", 1.0 - compressed_num_params / full_num_params
        )
        print("MACs reduction (%)       : ", 1.0 - compressed_macs / full_macs)
        print("BOPs reduction (%)       : ", 1.0 - compressed_bops / full_bops)
        print(
            "Weight size reduction (%): ",
            1.0 - compressed_weight_size / full_weight_size,
        )

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output - compressed_output))
        print("Maximum output difference: ", str(max_output_diff.item()))
        self.assertLessEqual(max_output_diff, 1e-4)
