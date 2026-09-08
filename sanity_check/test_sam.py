import unittest

# from PIL import Image
import torch
from backends import SamModel

# import requests
# from transformers import SamProcessor
from transformers import SamConfig

from only_train_once import OTO

OUT_DIR = "./cache"


class TestSam(unittest.TestCase):
    def test_sanity(self):
        config = SamConfig()
        # config.vision_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_hidden_layers = 1
        # config.mask_decoder_config.num_multimask_outputs = 1
        model = SamModel(config)

        dummy_input_encoder = torch.randn(1, 3, 1024, 1024)
        oto_encoder = OTO(model.vision_encoder, dummy_input_encoder)
        oto_encoder.visualize(view=False, out_dir=OUT_DIR, display_params=True)

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

        return
        # For test FLOP and param reductions.
        # full_flops = oto.compute_flops(in_million=True)['total']
        # full_num_params = oto.compute_num_params(in_million=True)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)


if __name__ == "__main__":
    unittest.main()
