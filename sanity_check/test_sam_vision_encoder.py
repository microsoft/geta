import os
import unittest

import torch
from backends import SamModel
from transformers import SamConfig

from only_train_once import OTO

OUT_DIR = "./cache"


class TestSamVisionEncoder(unittest.TestCase):
    def test_sanity(self):
        config = SamConfig()
        config.vision_config.num_hidden_layers = 1
        config.mask_decoder_config.num_hidden_layers = 1
        config.mask_decoder_config.num_multimask_outputs = 1
        model = SamModel(config)

        dummy_input_encoder = torch.randn(1, 3, 1024, 1024)
        oto_encoder = OTO(model.vision_encoder, dummy_input_encoder)
        oto_encoder.mark_unprunable_by_param_names(["pos_embed"])
        oto_encoder.mark_unprunable_by_param_names(["layers.0.attn.qkv.weight"])
        oto_encoder.mark_unprunable_by_param_names(["neck.layer_norm1.weight"])
        oto_encoder.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        oto_encoder.random_set_zero_groups()
        oto_encoder.construct_subnet(out_dir=OUT_DIR)

        full_sam_encoder = torch.load(oto_encoder.full_group_sparse_model_path)
        compressed_sam_encoder = torch.load(oto_encoder.compressed_model_path)

        full_output = full_sam_encoder(dummy_input_encoder)
        compressed_output = compressed_sam_encoder(dummy_input_encoder)

        print(dir(full_output))
        print(
            full_output.last_hidden_state.shape,
            compressed_output.last_hidden_state.shape,
        )

        max_output_diff = torch.max(
            torch.abs(
                full_output.last_hidden_state - compressed_output.last_hidden_state
            )
        )
        print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto_encoder.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto_encoder.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )

        # for name, param in compressed_sam_encoder.named_parameters():
        #     print(name, param.shape)


if __name__ == "__main__":
    unittest.main()
