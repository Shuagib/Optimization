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
n_epochs = 30

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

batch_size = 64

# Create data loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Hardware
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define the model Not with Momemtum
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


train_losses = []
test_losses = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss += criterion(output, labels).item()
        loss.backward()
        optimizer.step()
        step += 1
    train_loss /= len(train_loader.dataset)
    train_losses.append((step, train_loss))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append((step, test_loss))
    print(f'Baseline Test set: Average loss: {test_loss}, Train set: {train_loss} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
  





# Define the model SGD with Momentum
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_m = optim.SGD(model.parameters(), lr=0.001,momentum=0.09)


train_losses_m = []
test_losses_m = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_m = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_m.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_m += criterion(output, labels).item()
        loss.backward()
        optimizer_m.step()
        step += 1
    train_loss_m /= len(train_loader.dataset)
    train_losses_m.append((step, train_loss_m))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_m = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_m += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_m /= len(test_loader.dataset)
    test_losses_m.append((step, test_loss_m))
    print(f' SGDMomentum: Test set: Average loss: {test_loss_m}, Train set: {train_loss_m} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
  





# Define the model Adam
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)


train_losses_adam = []
test_losses_adam = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_adam = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_adam.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_adam += criterion(output, labels).item()
        loss.backward()
        optimizer_adam.step()
        step += 1
    train_loss_adam /= len(train_loader.dataset)
    train_losses_adam.append((step, train_loss_adam))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_adam = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_adam += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_adam /= len(test_loader.dataset)
    test_losses_adam.append((step, test_loss_adam))
    print(f' Adam: Test set: Average loss: {test_loss_adam}, Train set: {train_loss_adam} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')



# Define the model Adagrad
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_ag = optim.Adagrad(model.parameters(), lr=0.001)


train_losses_ag = []
test_losses_ag = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_ag = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_ag.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_ag += criterion(output, labels).item()
        loss.backward()
        optimizer_ag.step()
        step += 1
    train_loss_ag /= len(train_loader.dataset)
    train_losses_ag.append((step, train_loss_ag))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_ag = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_ag += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_ag /= len(test_loader.dataset)
    test_losses_ag.append((step, test_loss_ag))
    print(f' Adagrad: Test set: Average loss: {test_loss_ag}, Train set: {train_loss_ag} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')


# Define the model RMSProp
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_rms = optim.RMSprop(model.parameters(), lr=0.001, momentum= 0.9)


train_losses_rms = []
test_losses_rms = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_rms = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_rms.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_rms += criterion(output, labels).item()
        loss.backward()
        optimizer_rms.step()
        step += 1
    train_loss_rms /= len(train_loader.dataset)
    train_losses_rms.append((step, train_loss_rms))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_rms = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_rms += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_rms /= len(test_loader.dataset)
    test_losses_rms.append((step, test_loss_rms))
    print(f' RMSProp: Test set: Average loss: {test_loss_rms}, Train set: {train_loss_rms} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')


# Define the model AdaDelta
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_ad = optim.Adadelta(model.parameters(), lr=1.0)


train_losses_ad = []
test_losses_ad = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_ad = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_ad.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_ad += criterion(output, labels).item()
        loss.backward()
        optimizer_ad.step()
        step += 1
    train_loss_ad /= len(train_loader.dataset)
    train_losses_ad.append((step, train_loss_ad))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_ad = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_ad += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_ad /= len(test_loader.dataset)
    test_losses_ad.append((step, test_loss_ad))
    print(f' AdaDelta: Test set: Average loss: {test_loss_ad}, Train set: {train_loss_ad} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')


# Define the model NAdam
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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

model = LeNet5()

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_nadam = optim.NAdam(model.parameters(), lr=0.001,momentum_decay= 4e-3)


train_losses_nadam = []
test_losses_nadam = []

step = 0
for epoch in range(n_epochs):
    model.train()
    train_loss_nadam = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_nadam.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_nadam += criterion(output, labels).item()
        loss.backward()
        optimizer_nadam.step()
        step += 1
    train_loss_nadam /= len(train_loader.dataset)
    train_losses_nadam.append((step, train_loss_nadam))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model.eval()
    test_loss_nadam = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_loss_nadam += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_nadam /= len(test_loader.dataset)
    test_losses_nadam.append((step, test_loss_nadam))
    print(f' NAdam: Test set: Average loss: {test_loss_nadam}, Train set: {train_loss_nadam} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')


# Adjust layout and show the plot
plt.tight_layout()
# plt.show()
#plt.savefig('loss2.png')


#  plot train and test losses for SGD Baseline
train_steps, train_loss = zip(*train_losses)
test_steps, test_loss = zip(*test_losses)
# plot train and test losses for SGG Momentum
train_steps_m, train_loss_m = zip(*train_losses_m)
test_steps_m, test_loss_m = zip(*test_losses_m)
# plot train and test losses  for Adam Momentum
train_steps_a, train_loss_a = zip(*train_losses_adam)
test_steps_a, test_loss_a = zip(*test_losses_adam)
# plot train and test losses for Adagrad
train_steps_ag, train_loss_ag = zip(*train_losses_ag)
test_steps_ag, test_loss_ag = zip(*test_losses_ag)
# plot train and test losses for RMSProp
train_steps_rms, train_loss_rms = zip(*train_losses_rms)
test_steps_rms, test_loss_rms = zip(*test_losses_rms)
# plot train and test losses for AdaDelta
train_steps_ad, train_loss_ad = zip(*train_losses_ad)
test_steps_ad, test_loss_ad = zip(*test_losses_ad)
# plot train and test losses for NAdam
train_steps_nadam, train_loss_nadam = zip(*train_losses_nadam)
test_steps_nadam, test_loss_nadam = zip(*test_losses_nadam)





# Plot the training loss  and validation alltogether 
plt.plot(range(1,n_epochs+1), train_loss, label='Training Loss Baseline')
plt.plot(range(1,n_epochs+1), test_loss, label='Test Loss Baseline', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_m, label='Training Loss SGD Momentum')
plt.plot(range(1,n_epochs+1), test_loss_m, label='Test Loss SGD Momentum', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_a, label='Training Loss Adam')
plt.plot(range(1,n_epochs+1), test_loss_a, label='Test Loss Adam', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_ag, label='Training Loss Adagrad')
plt.plot(range(1,n_epochs+1), test_loss_ag, label='Test Loss Adagrad', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_rms, label='Training Loss RMSProp')
plt.plot(range(1,n_epochs+1), test_loss_rms, label='Test Loss RMSProp', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_ad, label='Training Loss AdaDelta')
plt.plot(range(1,n_epochs+1), test_loss_ad, label='Test Loss AdaDelta', linestyle='--')
plt.plot(range(1,n_epochs+1), train_loss_nadam, label='Training Loss NAdam')
plt.plot(range(1,n_epochs+1), test_loss_nadam, label='Test Loss NAdam', linestyle='--')


plt.title('Momentum: Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
#Creating a box that isn't too large
plt.legend(fontsize=7, ncols=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Compare_Momentum_Algorithms.png', bbox_inches='tight')
plt.show()


