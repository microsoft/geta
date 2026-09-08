import os
import unittest

import torch
from backends import DiffIRS3SNPE

from only_train_once import OTO

OUT_DIR = "./cache"

"""
feat.shape torch.Size([1, 48, 64, 64])
k_v.shape torch.Size([1, 256])
inp_enc_level1.shape torch.Size([1, 64, 64, 64])
out_enc_level1.shape torch.Size([1, 64, 64, 64])
inp_enc_level2.shape torch.Size([1, 128, 32, 32])
out_enc_level2.shape torch.Size([1, 128, 32, 32])
inp_enc_level3.shape torch.Size([1, 256, 16, 16])
out_enc_level3.shape torch.Size([1, 256, 16, 16])
inp_enc_level4.shape torch.Size([1, 512, 8, 8])
latent.shape torch.Size([1, 512, 8, 8])

inp_dec_level3.shape torch.Size([1, 256, 16, 16])
inp_dec_level3.shape after cat torch.Size([1, 512, 16, 16])
inp_dec_level3.shape after chan red torch.Size([1, 256, 16, 16])
out_dec_level3.shape torch.Size([1, 256, 16, 16])

inp_dec_level2.shape torch.Size([1, 128, 32, 32])
inp_dec_level2.shape after cat torch.Size([1, 256, 32, 32])
inp_dec_level2.shape after chan red torch.Size([1, 128, 32, 32])
out_dec_level2.shape torch.Size([1, 128, 32, 32])

inp_dec_level1.shape torch.Size([1, 64, 64, 64])
inp_dec_level1.shape after cat torch.Size([1, 128, 64, 64])
out_dec_level1.shape torch.Size([1, 128, 64, 64])
out_dec_level1.shape after refine torch.Size([1, 128, 64, 64])
out_dec_level1.shape after tail torch.Size([1, 3, 256, 256])
"""


class TestDITSRUpPath(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):

        model = DiffIRS3SNPE(
            num_blocks=[2, 1, 1, 1],  # [13,1,1,1],
            # num_refinement_blocks=1
        )
        out_enc_level1 = torch.randn(1, 64, 64, 64)
        out_enc_level2 = torch.randn(1, 128, 32, 32)
        out_enc_level3 = torch.randn(1, 256, 16, 16)
        latent_up_path = torch.randn(1, 512, 8, 8)
        k_v = torch.randn(1, 256)

        oto = OTO(
            model.G.up_path,
            (out_enc_level1, out_enc_level2, out_enc_level3, latent_up_path, k_v),
        )
        # oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(
            out_enc_level1, out_enc_level2, out_enc_level3, latent_up_path, k_v
        )
        compressed_output = compressed_model(
            out_enc_level1, out_enc_level2, out_enc_level3, latent_up_path, k_v
        )

        if isinstance(full_output, tuple):
            for full_out, compress_out in zip(full_output, compressed_output):
                max_output_diff = 0.0
                if full_out.shape == compress_out.shape:
                    max_output_diff = torch.max(torch.abs(full_out - compress_out))
                else:
                    full_out_tmp = full_out.squeeze(0)
                    full_out_tmp = full_out_tmp.view(full_out_tmp.shape[0], -1)
                    p_norm = torch.norm(full_out_tmp, dim=1)
                    max_output_diff = torch.max(
                        torch.abs(full_out[:, p_norm != 0.0] - compress_out)
                    )
                print("Maximum output difference " + str(max_output_diff.item()))
        else:
            max_output_diff = torch.max(torch.abs(full_output - compressed_output))
            print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )


class TestDITSRDownPath(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):

        model = DiffIRS3SNPE(
            num_blocks=[2, 1, 1, 1]  # [13,1,1,1],
        )
        feat = torch.rand(1, 48, 56, 56)
        latent = torch.randn(1, 256)

        oto = OTO(model.G.down_path, (feat, latent))
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        # prunable_param_names = [
        #     'encoder_level3.0.ffn.project_in.weight',
        #     'latent.0.ffn.project_in.weight',
        #     'encoder_level2.0.ffn.project_in.weight',
        #     'encoder_level1.0.ffn.project_in.weight',

        #     'encoder_level2.0.ffn.kernel.0.weight',
        #     'encoder_level3.0.attn.kernel.0.weight',
        #     'encoder_level1.0.ffn.kernel.0.weight',
        #     'latent.0.attn.kernel.0.weight'
        #     # 'encoder_level1.0.ffn.kernel.0.weight'
        #     # # 'encoder_level1.0.attn.kernel.0.weight'
        # ]
        # for node_group in oto._graph.node_groups.values():
        #     if any([True if prunable_param_name in node_group.param_names else False for prunable_param_name in prunable_param_names]):
        #         node_group.is_prunable = True
        #     else:
        #         node_group.is_prunable = False

        # post_process_chunk_node(oto._graph)
        # oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        oto.mark_unprunable_by_param_names(
            [
                "down1_2.body.0.weight",
                "down2_3.body.0.weight",
                "down3_4.body.0.weight",
            ]
        )
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(feat, latent)
        compressed_output = compressed_model(feat, latent)

        if isinstance(full_output, tuple):
            for full_out, compress_out in zip(full_output, compressed_output):
                max_output_diff = 0.0
                if full_out.shape == compress_out.shape:
                    max_output_diff = torch.max(torch.abs(full_out - compress_out))
                else:
                    full_out_tmp = full_out.squeeze(0)
                    full_out_tmp = full_out_tmp.view(full_out_tmp.shape[0], -1)
                    p_norm = torch.norm(full_out_tmp, dim=1)
                    max_output_diff = torch.max(
                        torch.abs(full_out[:, p_norm != 0.0] - compress_out)
                    )
                print("Maximum output difference " + str(max_output_diff.item()))
        else:
            max_output_diff = torch.max(torch.abs(full_output - compressed_output))
            print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )


class TestDITSR(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):

        model = DiffIRS3SNPE()

        lq = torch.rand(1, 3, 256, 256)
        latent = torch.rand(1, 256)

        oto = OTO(model.G, (lq, latent))
        oto.mark_unprunable_by_param_names(
            [
                "down_path.down1_2.body.0.weight",
                "down_path.down2_3.body.0.weight",
                "down_path.down3_4.body.0.weight",
            ]
        )

        # oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)
        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)

        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(lq, latent)
        compressed_output = compressed_model(lq, latent)

        if isinstance(full_output, tuple):
            for full_out, compress_out in zip(full_output, compressed_output):
                max_output_diff = 0.0
                if full_out.shape == compress_out.shape:
                    max_output_diff = torch.max(torch.abs(full_out - compress_out))
                else:
                    full_out_tmp = full_out.squeeze(0)
                    full_out_tmp = full_out_tmp.view(full_out_tmp.shape[0], -1)
                    p_norm = torch.norm(full_out_tmp, dim=1)
                    max_output_diff = torch.max(
                        torch.abs(full_out[:, p_norm != 0.0] - compress_out)
                    )
                print("Maximum output difference " + str(max_output_diff.item()))
        else:
            max_output_diff = torch.max(torch.abs(full_output - compressed_output))
            print("Maximum output difference " + str(max_output_diff.item()))
        # self.assertLessEqual(max_output_diff, 1e-4)
        full_model_size = os.stat(oto.full_group_sparse_model_path)
        compressed_model_size = os.stat(oto.compressed_model_path)
        print("Size of full model     : ", full_model_size.st_size / (1024**3), "GBs")
        print(
            "Size of compress model : ",
            compressed_model_size.st_size / (1024**3),
            "GBs",
        )
