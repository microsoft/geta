import unittest

import torch
from backends import SamModel
from transformers import SamConfig

from only_train_once import OTO

OUT_DIR = "./cache"


class TestSamMaskDecoder(unittest.TestCase):
    def test_sanity(self):
        config = SamConfig()
        # config.vision_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_multimask_outputs = 1
        model = SamModel(config)

        dummy_input_decoder = (
            torch.randn(1, 256, 64, 64),
            torch.randn(1, 256, 64, 64),
            torch.randn(1, 1, 1, 256),
            torch.randn(1, 256, 64, 64),
            True,
            None,
            None,
            None,
        )
        oto_decoder = OTO(model.mask_decoder, dummy_input_decoder)
        oto_decoder.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        oto_decoder.random_set_zero_groups()
        oto_decoder.construct_subnet(out_dir=OUT_DIR)


if __name__ == "__main__":
    unittest.main()
