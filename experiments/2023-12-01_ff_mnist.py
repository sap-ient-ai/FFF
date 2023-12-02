# based on https://github.com/AquibPy/Pytorch/blob/master/MNIST%20Using%20ANN%20on%20GPU%20with%20Pytorch.ipynb

from torch import nn, functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from random import sample
import nevergrad as ng


import torchvision
import torchvision.transforms as transforms

# Transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

device = torch.device('cuda')
# Data loaders
trainloader = torch.utils.data.DataLoader(trainset, num_workers=8, batch_size=256, shuffle=True,  pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset , num_workers=8, batch_size=256, shuffle=False, pin_memory=True)

NEPOCH = 5

#region CUDA
def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)
    

#endregion CUDA
trainloader = DeviceDataLoader(trainloader, device)
testloader = DeviceDataLoader(trainloader, device)

trainloader = list(trainloader)
testloader = list(testloader)

class FFF_wrapper(nn.Module):
    def __init__(self, parent, size):
        super(FFF_wrapper, self).__init__()
        self._parent = (parent,)
        self.size = size
        nn_size = parent.fc1.out_features
        self.slice = torch.nn.Parameter(
            torch.tensor(
                sample(range(0, nn_size), k=size),
                dtype=torch.int
            ),
            requires_grad=False
        )
        self.slice.cuda()
        

    def forward(self, x):
        x = x.view(-1, 28*28)
        fc1_bias, fc1_weight, fc2_weight, fc2_bias = self._parent[0].get_slice(self.slice)
        
        fc1 = nn.Linear(28*28, self.size).cuda()
        fc2 = nn.Linear(self.size, 10).cuda()

        fc1.bias = fc1_bias
        fc1.weight = fc1_weight
        
        fc2.weight = fc2_weight
        fc2.bias = fc2_bias

        x = F.relu(fc1(x))
        x = fc2(x)
        return x



# Neural network architecture
class Net(nn.Module):
    def __init__(self, size=32):
        super(Net, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(28*28, size)
        self.fc2 = nn.Linear(size, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def calibrate(self, num_nodes):
        # "Overall performance" -> Similar to training
        # "Best use" -> Considering whole input space, which node combination better separate whole space?
        # "Best resolution" -> Considering "close regions" of input space, which node combination is able to discern them?
        nn_size = self.fc1.out_features
        assert num_nodes < nn_size

        slice_idx = sample(range(0, nn_size),k=num_nodes)
        
        # allocate:
        #fc1 = nn.Linear(28*28, num_nodes)
        #fc2 = nn.Linear(num_nodes, 10)

        new_net = Net(num_nodes)
        new_net.fc1.bias = torch.nn.Parameter(self.fc1.bias[slice_idx])
        new_net.fc1.weight = torch.nn.Parameter(self.fc1.weight[slice_idx])

        new_net.fc2.weight = torch.nn.Parameter(self.fc2.weight[:, slice_idx])
        new_net.fc2.bias = torch.nn.Parameter(self.fc2.bias)
        return new_net
    
    def get_slice(self, slice_idx):
        fc1_bias = torch.nn.Parameter(self.fc1.bias[slice_idx])
        fc1_weight = torch.nn.Parameter(self.fc1.weight[slice_idx])
        fc2_weight = torch.nn.Parameter(self.fc2.weight[:, slice_idx])
        fc2_bias = torch.nn.Parameter(self.fc2.bias)
        return fc1_bias, fc1_weight, fc2_weight, fc2_bias


    def fff_forward(self, x, n):
        import math
        tree_depth = math.floor(math.log2(self.fc1.out_features))
        assert n <= tree_depth
        self.fc1
        pass


FF_net = Net()
FF_net = FF_net.cuda()


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

# https://paperswithcode.com/method/adamw
# https://arxiv.org/abs/1711.05101
# interesting:
# https://arxiv.org/abs/2202.00089
# https://arxiv.org/abs/2302.00195
# https://arxiv.org/abs/2112.06125



def train(network, lr=0.001):
    optimizer = optim.AdamW(network.parameters(), lr=lr, foreach=True) #, momentum=0.9)
    # Training the network
    for epoch in tqdm(range(NEPOCH)):  # loop over the dataset multiple times
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Running loss {running_loss}')

    print('Finished Training')

train(FF_net)


def evaluate(network):
    # Testing the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
evaluate(FF_net)


fff = FFF_wrapper(FF_net, 6)

ex = trainloader[0][0][0]

if __name__ == '__main__':
    from IPython import embed
    embed(colors='neutral')
