import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

#Global epochs
n_epochs = 10

seed = 42
torch.manual_seed(seed)
#random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))

print("datatype of the 1st training sample: ", train_dataset[0][0].type())
print("size of the 1st training sample: ", train_dataset[0][0].size())
#Taking a average base size
batch_size = 64

# Create data loaders.


######################################################

batch_size_mini = 5 #Creating a small batch size

# Create data loaders.
train_loader_mini = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size_mini, shuffle=True)
test_loader_mini = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader_mini:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#Mini
class LeNet5_mini(nn.Module):
    def __init__(self):
        super(LeNet5_mini, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

model = LeNet5_mini()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_mini = optim.SGD(model.parameters(), lr=0.01)


train_losses_mini = []
test_losses_mini = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_mini = 0
    for i, (images, labels) in enumerate(train_loader_mini):
        optimizer_mini.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_mini += criterion(output, labels).item()
        loss.backward()
        optimizer_mini.step()
        step += 1
    train_loss_mini /= len(train_loader_mini)
    train_losses_mini.append((step, train_loss_mini))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_mini = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader_mini:
            output = model(images)
            test_loss_mini += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_mini /= len(test_loader_mini)
    test_losses_mini.append((step, test_loss_mini))
    print(f'Mini SGD Test set: Average loss: {test_loss_mini}, Train set: {train_loss_mini} \
        Accuracy: {correct}/{len(test_loader_mini.dataset)} ({100. * correct / len(test_loader_mini.dataset)}%)')
  




transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))

print("datatype of the 1st training sample: ", train_dataset[0][0].type())
print("size of the 1st training sample: ", train_dataset[0][0].size())
#Taking a average base size
batch_size = 64




# Create data loaders.
train_loader_fSG = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= len(train_dataset), shuffle=True) #Creating the batch size as the whole training data
test_loader_fSG = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader_fSG:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Define the model Not with Momemtum
class LeNet5_GD(nn.Module):
    def __init__(self):
        super(LeNet5_GD, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

model = LeNet5_GD()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 1)


train_losses_fGS = []
test_losses_fGS = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_fGS = 0
    for i, (images, labels) in enumerate(train_loader_fSG):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_fGS += criterion(output, labels).item()
        loss.backward()
        optimizer.step()
        step += 1
    train_loss_fGS /= len(train_loader_fSG)
    train_losses_fGS.append((step, train_loss_fGS))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_fSG = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader_fSG:
            output = model(images)
            test_loss_fSG += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_fSG /= len(test_loader_fSG)
    test_losses_fGS.append((step, test_loss_fSG))
    print(f'full batch size SGD Test set: Average loss: {test_loss_fSG}, Train set: {train_loss_fGS} \
        Accuracy: {correct}/{len(test_loader_fSG.dataset)} ({100. * correct / len(test_loader_fSG.dataset)}%)')
  




train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# Define the model Not with Momemtum
class LeNet5_SGD(nn.Module):
    def __init__(self):
        super(LeNet5_SGD, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

model = LeNet5_SGD()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_SGD = optim.SGD(model.parameters(), lr=0.1)


train_losses = []
test_losses = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_SGD = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_SGD.zero_grad()
        output = model(images)
        loss_SGD = criterion(output, labels)
        train_loss_SGD += criterion(output, labels).item()
        loss_SGD.backward()
        optimizer_SGD.step()
        step += 1
    train_loss_SGD /= len(train_loader)
    train_losses.append((step, train_loss_SGD))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss_SGD.item()}')

    model.eval()
    test_loss_SGD = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_SGD += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_SGD /= len(test_loader)
    test_losses.append((step, test_loss_SGD))
    print(f'Baseline SGD Test set: Average loss: {test_loss_SGD}, Train set: {train_loss_SGD} \
        Accuracy: {correct}/{len(test_loader)} ({100. * correct / len(test_loader.dataset)}%)')
  





#  plot train and test losses for SGD Baseline
train_steps, train_loss_SGD = zip(*train_losses)
test_steps, test_loss_SGD = zip(*test_losses)

# plot train and test losses for GD
train_steps_m, train_loss_m = zip(*train_losses_fGS)
test_steps_m, test_loss_m = zip(*test_losses_fGS)

#Plotting train and test losses for mini SGD
train_steps, train_loss_min = zip(*train_losses_mini)
test_steps, test_loss_min = zip(*test_losses_mini)


#Plotting SGD
plt.plot(range(1,n_epochs+1), test_loss_SGD, label='Test Loss  SGD', linestyle='--', color = 'blue')
plt.plot(range(1,n_epochs+1), train_loss_SGD, label='Training Loss SGD',color='blue')

#Plotting GD
plt.plot(range(1,n_epochs+1), test_loss_m, label='Test Loss basis GD', linestyle='--', color = 'red')
plt.plot(range(1,n_epochs+1), train_loss_m, label='Training Loss GD', color ='red')

#Plotting Mini SGD
plt.plot(range(1,n_epochs+1), test_loss_min, label='Test Loss Mini SGD', linestyle='--', color = 'green')
plt.plot(range(1,n_epochs+1), train_loss_min, label='Training Loss Mini SGD', color ='green')


plt.title(' SGD vs Gradient descent vs Mini  : Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
#Creating a box that isn't too large
plt.legend(fontsize=7, ncols=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('SGDvsfullSGD.png', bbox_inches='tight')
plt.show()

