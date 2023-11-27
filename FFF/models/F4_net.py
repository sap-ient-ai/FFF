import torch
from torch import nn
from FFF.experiments.ffff import F4


class F4_mnist(nn.Module):
    def __init__(self, in_features, hidden_features, out_classes, num_trees):
        super(F4_mnist, self).__init__()
        self.fc1 = F4(
            in_features=in_features,
            out_features=hidden_features,
            num_trees=num_trees,
            use_depth_fair=True,
        )
        self.fc2 = F4(
            in_features=hidden_features,
            out_features=out_classes,
            num_trees=num_trees,
            use_depth_fair=True,
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        y_hat = self.fc2(x)
        return y_hat
