import os
import unittest

import torch
from ultralytics import YOLO

from only_train_once import OTO

OUT_DIR = "./cache"

"""
C2f
Bottleneck
Module definition
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
"""


class TestYolov8(unittest.TestCase):
    def test_sanity(self, dummy_input=torch.rand(1, 3, 224, 224)):
        # Load a model
        # model = YOLO("yolov8n.yaml")  # build a new model from scratch
        yolov8_model = YOLO(
            "yolov8n.pt"
        )  # load a pretrained model (recommended for training)

        for name, param in yolov8_model.model.named_parameters():
            # print(name, param.shape, param.requires_grad)
            if "running_mean" not in name:
                param.requires_grad = True
        # print(type(yolov8_model.model))
        oto = OTO(yolov8_model.model, dummy_input)
        oto.mark_unprunable_by_param_names(
            [
                "model.22.dfl.conv.weight",
                "model.22.cv3.2.2.weight",
                "model.22.cv2.2.2.weight",
                "model.22.cv2.1.2.weight",
                "model.22.cv3.1.2.weight",
                "model.22.cv3.0.2.weight",
                "model.22.cv2.0.2.weight",
            ]
        )
        # for node_group in oto._graph.node_groups.values():
        #     for node in node_group:
        #         if node.op_name == 'slice':
        #             node_group.is_prunable = False
        oto.visualize(view=False, out_dir=OUT_DIR, display_params=True)

        oto.random_set_zero_groups()
        oto.construct_subnet(out_dir=OUT_DIR)
        full_model = torch.load(oto.full_group_sparse_model_path)
        compressed_model = torch.load(oto.compressed_model_path)

        full_output = full_model(dummy_input)
        compressed_output = compressed_model(dummy_input)

        max_output_diff = torch.max(torch.abs(full_output[0] - compressed_output[0]))
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
