import unittest

import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model

OUT_DIR = "./cache"


class TestQBert(unittest.TestCase):
    def test_sanity(self, dummy_input=None):
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=OUT_DIR,
        )
        config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_hidden_layers=2,  # Comment out for full Bert model
            cache_dir=OUT_DIR,
        )
        model = AutoModelForQuestionAnswering.from_config(config)
        q_model = model_to_quantize_model(model)
        text = (
            "This is a test sentence of a very long string and random wording that is used to test dolly model."
            * 7
        )
        input_data = tokenizer(text, return_tensors="pt").input_ids

        oto = OTO(q_model, dummy_input=(input_data,), strict_out_nodes=False)

        # Exclude emebdding
        oto.mark_unprunable_by_param_names(["bert.embeddings.word_embeddings.weight"])
        oto.visualize(
            view=False, out_dir=OUT_DIR, display_flops=False, display_params=True
        )
        oto.random_set_zero_groups(target_group_sparsity=0.9)
        oto.construct_subnet(
            export_huggingface_format=False,
            export_float16=False,
            full_group_sparse_model_dir=OUT_DIR,
            compressed_model_dir=OUT_DIR,
        )

        text_1 = (
            "This is a test sentence of a very long string and random wording that is used to test dolly model."
            * 7
        )
        input_data_1 = tokenizer(text_1, return_tensors="pt").input_ids

        text_2 = (
            "This is a good test sentence of a pretty short string and wording that is used to test dolly model."
            * 7
        )
        input_data_2 = tokenizer(text_2, return_tensors="pt").input_ids

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)
        oto_compressed = OTO(compressed_model, dummy_input=(input_data_1,))
        full_output_1 = full_model(input_data_1.to(full_model.device))
        full_output_2 = full_model(input_data_2.to(full_model.device))
        compressed_output_1 = compressed_model(input_data_1.to(compressed_model.device))
        compressed_output_2 = compressed_model(input_data_2.to(compressed_model.device))
        max_output_diff_1 = torch.max(full_output_1[0] - compressed_output_1[0]).item()
        max_output_diff_2 = torch.max(full_output_2[0] - compressed_output_2[0]).item()

        full_macs = oto.compute_macs(in_million=True, layerwise=True)
        full_bops = oto.compute_bops(in_million=True, layerwise=True)
        full_num_params = oto.compute_num_params(in_million=True)
        full_weight_size = oto.compute_weight_size(in_million=True)

        compressed_macs = oto_compressed.compute_macs(in_million=True, layerwise=True)
        compressed_bops = oto_compressed.compute_bops(in_million=True, layerwise=True)
        compressed_num_params = oto_compressed.compute_num_params(in_million=True)
        compressed_weight_size = oto_compressed.compute_weight_size(in_million=True)

        print(f"Full MACs for QBert       : {full_macs['total']} M MACs")
        print(f"Full BOPs for QBert       : {full_bops['total']} M BOPs")
        print(f"Full num params for QBert : {full_num_params} M params")
        print(f"Full weight size for QBert: {full_weight_size['total']} M bits")

        print(f"Compressed MACs for QBert       : {compressed_macs['total']} M MACs")
        print(f"Compressed BOPs for QBert       : {compressed_bops['total']} M BOPs")
        print(f"Compressed num params for QBert : {compressed_num_params} M params")
        print(
            f"Compressed weight size for QBert: {compressed_weight_size['total']} M bits"
        )

        print(f"Maximum output difference under the same inputs: {max_output_diff_1}")
        print(f"Maximum output difference under the same inputs: {max_output_diff_2}")

        self.assertLessEqual(max_output_diff_1, 1e-4)
        self.assertLessEqual(max_output_diff_2, 1e-4)
