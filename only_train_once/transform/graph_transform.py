# import imp
import re

from . import ge

# from only_train_once.operation import Operator
# from only_train_once.graph import Node


class Rename:
    def __init__(self, op=None, name=None, to=None):
        assert op or name, "Either op or name must be provided"
        assert not (op and name), "Either op or name should be provided, but not both"
        assert bool(to), "The to parameter is required"
        self.to = to
        self.op = re.compile(op) if op else None
        self.name = re.compile(name) if name else None

    def apply(self, graph):
        for i, node in enumerate(graph.nodes.values()):
            if self.op:
                node.op_name = self.op.sub(self.to, node.op_name)
            if self.name is None:
                node.op_name = str(node.op_name)
            else:
                node.op_name = self.name.sub(self.to, node.op_name)


class Fold:
    def __init__(self, pattern, to, name=None):
        # TODO: validate that op and name are valid
        self.pattern = ge.GEParser(pattern).parse()
        self.to = to
        self.name = name

    def apply(self, graph):
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # Replace pattern with new node
            if self.to == "__first__":
                combo = matches[0]
            elif self.to == "__last__":
                combo = matches[-1]
            else:
                # find the most bottom child
                outputs = set()
                match_ids = [node.id for node in matches]
                for match_node in matches:
                    for outgoing_node in graph.outgoing(match_node):
                        if outgoing_node.id not in match_ids:
                            outputs.add(outgoing_node)
                # combine operators
                combo_op = matches[0].op
                for i in range(1, len(matches)):
                    combo_op += matches[i].op
                combo_op.name = self.to or self.pattern
                combo = Node(
                    id=graph.sequence_id(),
                    op=combo_op,
                    output_shape=matches[-1].output_shape,
                    outputs=list(outputs),
                )  # TODO, check bugs
                combo._caption = "/".join(filter(None, [l.caption for l in matches]))
            graph.replace(matches, combo)


# class FoldPixelUnShuffle():
#     def __init__(self, pattern, to, name=None):
#         # TODO: validate that op and name are valid
#         self.pattern = ge.GEParser(pattern).parse()
#         self.to = to
#         self.name = name

#     def apply(self, graph):
#         while True:
#             matches, _ = graph.search(self.pattern)
#             print(self.pattern)
#             # print(matches)
#             for node in matches:
#                 print(node, node.input_shape, node.output_shape)
#             if not matches:
#                 break
#             assert len(matches) == 5
#             reshape_first_node = matches[1]
#             conv_second_node = matches[4]
#             upscale_ratio = conv_second_node.input_shape[1] // reshape_first_node.input_shape[1]
#             if upscale_ratio > 1:
#                 op = Operator(_type=self.to, cfg_params={'upscale_ratio': upscale_ratio})
#                 # node = Node(id=, op_name=self.to, op=op, inputs=inputs, outputs=outputs, output_shape=output_shape)
#                 # graph.replace(matches, combo)


class ConvBNFuse:
    def __init__(self, pattern, to, name=None):
        self.pattern = ge.GEParser(pattern).parse()
        self.to = to
        self.name = name

    def apply(self, graph):
        graph.fused_conv_bns = list()
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            for match_node in matches:
                match_node._skip_pattern_search = True
            graph.fused_conv_bns.append(matches)


# PyTorch Graph Transforms
FRAMEWORK_TRANSFORMS = [
    Rename(op=r"onnx::(.*)", to=r"\1"),
    Rename(op=r"gemm", to=r"linear"),
    Rename(op=r"batchnormalization", to="batchnorm"),
    # Tackle qualified pixelunshuffle
    # FoldPixelUnShuffle('conv > reshape > transpose > reshape > conv', to='spacetodepth')
]

# CONV_BN_FUSE = ConvBNFuse("conv > batchnorm", "convbn")
