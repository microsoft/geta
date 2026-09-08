"""
Implementation of multilayer perceptron
"""

from torch import nn


class MLP(nn.Module):
    def __init__(
        self, hidden_size1, intermediate_size1, intermediate_size2, hidden_size2
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(
            hidden_size1,
            intermediate_size1,
            bias=True,
        )

        self.linear2 = nn.Linear(
            intermediate_size1,
            intermediate_size2,
            bias=True,
        )

        self.linear3 = nn.Linear(
            intermediate_size2,
            hidden_size2,
            bias=True,
        )

    def forward(self, x):
        x1 = self.linear1(self.flatten(x))
        x2 = self.linear2(nn.functional.relu(x1))
        x3 = self.linear3(nn.functional.relu(x2))

        return x3
