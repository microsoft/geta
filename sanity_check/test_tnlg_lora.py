import os
import unittest

import torch
from backends import TNLG, TNLGTokenizer
from peft_lora.lora_model import LoraConfig, LoraModel

from only_train_once import OTO

OUT_DIR = "./cache"


class TestTNLGLoRA(unittest.TestCase):
    def test_sanity(self, dummy_input=None):
        n_layers = 4
        hidden_size = 4096
        n_heads = 32
        max_seq_len = 4096
        vocab_size = 100352

        device = torch.device("cpu")

        model = TNLG(
            n_layers=n_layers,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
            seq_len=max_seq_len,
            device=device,
        )
        tokenizer = TNLGTokenizer()

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["to_4h", "to_h", "key_w", "value_w", "query_w"],
            lora_dropout=0.05,
            bias="none",
        )
        model = LoraModel(model, config)

        text = (
            "This is a test sentence of a very long string and random wording that is used to test dolly model."
            * 7
        )
        dummy_input = torch.IntTensor(tokenizer.tokenize(text)).to(device)

        oto = OTO(model, dummy_input, strict_out_nodes=True)

        oto.visualize(out_dir=OUT_DIR)
        oto.random_set_zero_groups()

        oto.construct_subnet(
            merge_lora_to_base=True,
            export_huggingface_format=False,
            export_float16=False,
            full_group_sparse_model_dir=OUT_DIR,
            compressed_model_dir=OUT_DIR,
        )

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_model.eval()
        compressed_model.eval()

        text_1 = (
            "This is a test sentence of a very long string and random wording that is used to test dolly model."
            * 7
        )
        text_2 = "How old are you? How are you? I am fine, and you" * 7
        tokens_1 = torch.IntTensor(tokenizer.tokenize(text_1)).to(device)
        tokens_2 = torch.IntTensor(tokenizer.tokenize(text_2)).to(device)

        full_output = full_model(tokens_1)
        compressed_output = compressed_model(tokens_1)

        max_output_diff_1 = torch.max(full_output[0] - compressed_output[0]).item()
        print("Maximum output difference under the same inputs:")
        print(max_output_diff_1)

        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )

        self.assertLessEqual(max_output_diff_1, 6.0)
