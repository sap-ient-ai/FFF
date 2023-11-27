import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


# MNIST dataset
trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


# training hyperparameters
Batch_Size = 64
NEPOCH = 10
lr = 0.001

# Data loaders
trainloader = data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True)
testloader = data.DataLoader(testset, batch_size=Batch_Size, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_ = optim.AdamW


def training_loop(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer = optimizer_,
    trainloader: data.DataLoader = trainloader,
    epochs: int = NEPOCH,
    lr: float = lr,
):
    # connect optimizer and model
    optimizer = optimizer_(net.parameters(), lr=lr)  # , momentum=0.9)

    epoch_losses = []
    # Training the network
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        epoch_losses.append(running_loss)
    print("Finished Training")
    return epoch_losses


def run_test(net: torch.nn.Module, testloader: data.DataLoader = testloader):
    # Testing the network on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
    return correct, total
