import unittest
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

"""
Quantization test cases
"""
# from test_qmlp import TestQMLP
# from test_qvgg7bn import TestQVGG7BN
# from test_qresnet18 import TestQResNet18
# from test_qresnet20 import TestQResNet20
# from test_qresnet56 import TestQResNet56
# from test_qresnet50 import TestQResNet50
# from test_qbert import TestQBert
# from test_qcarn import TestQCARN
# from test_qyolov5 import TestQYolov5
# from test_qsimplevit import TestQSimpleViT
# from test_qphi2 import TestQPhi2
# from test_qdeit import TestQDeiT
from test_qswin import TestQSwin

# from test_simplevit import TestSimpleViT
# from test_vit import TestViT
# from test_phi2 import TestPhi2
# from test_pvt import TestPVT
# from test_deit import TestDeiT
# from test_swin import TestSwin

OUT_DIR = './cache'

os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == '__main__':
    unittest.main()