import torch
from torch import nn
from FFF.fff import FFF


# Neural network architecture
class F3_net(nn.Module):
    def __init__(self):
        super(F3_net, self).__init__()
        self.fc1 = FFF(nIn=28 * 28, nOut=500)
        self.fc2 = FFF(nIn=500, nOut=10)
        # self.fc1 = FFF(nIn=28*28, nOut=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y_hat = self.fc2(torch.relu(self.fc1(x)))
        # y_hat = self.fc1(x)
        return y_hat
