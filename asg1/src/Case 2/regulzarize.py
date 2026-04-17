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

batch_size = 64

# Create data loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break





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

model_SGD = LeNet5()

count_parameters(model_SGD)
fc12_params = [p for name, p in model_SGD.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_SGD.parameters(), lr=0.2)
swa_model_swg  = optim.swa_utils.AveragedModel(model_SGD)
scheduler_SWG = optim.lr_scheduler.ExponentialLR(optimizer,gamma= 0.9)
scheduler_SWG2 = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10)

train_losses = []
test_losses = []
acc_sgd = []
step = 0
for epoch in range(n_epochs):
    model_SGD.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model_SGD(images)
        loss = criterion(output, labels)
        train_loss += criterion(output, labels).item()
        loss.backward()
        optimizer.step()
        step += 1
    #scheduler_SWG.step()
    #scheduler_SWG2.step()
    swa_model_swg.update_parameters(model_SGD)
    train_loss /= len(train_loader.dataset)
    train_losses.append((step, train_loss))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model_SGD.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model_SGD(images)
            test_loss += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append((step, test_loss))
    print(f'Baseline Test set: Average loss: {test_loss}, Train set: {train_loss} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
    acc_sgd.append(100. * correct / len(test_loader.dataset))
  






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
optimizer_adam = optim.AdamW(model.parameters(), lr=0.0001)
swa_model_adam  = optim.swa_utils.AveragedModel(model)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max= 30)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer_adam,gamma= 0.7)

train_losses_adam = []
test_losses_adam = []
acc_ = []
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
    #scheduler.step()
    #scheduler2.step()
    #swa_model_adam.update_parameters(model)
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
    acc = 100. * correct / len(test_loader.dataset)
    test_losses_adam.append((step, test_loss_adam))
    print(f' Adam: Test set: Average loss: {test_loss_adam}, Train set: {train_loss_adam} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({acc}%)')
    acc_.append(acc)








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

model_na = LeNet5()

count_parameters(model_na)
fc12_params = [p for name, p in model_na.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer_nadam = optim.NAdam(model_na.parameters(), lr= 0.0001,momentum_decay= 0.01) 
swa_model  = optim.swa_utils.AveragedModel(model_na)
scheduler_Nadam = optim.lr_scheduler.ExponentialLR(optimizer_nadam,gamma= 0.8)
scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_nadam,T_max=10 )
scheduler4 = optim.lr_scheduler.PolynomialLR(optimizer_nadam,total_iters= 10)
train_losses_nadam = []
test_losses_nadam = []
ac_na = []
step = 0
for epoch in range(n_epochs):
    model_na.train()
    train_loss_nadam = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer_nadam.zero_grad()
        output = model_na(images)
        loss = criterion(output, labels)
        train_loss_nadam += criterion(output, labels).item()
        loss.backward()
        optimizer_nadam.step()
        step += 1
    #scheduler3.step()
    #scheduler_Nadam.step()
    #scheduler4.step()
    #swa_model.update_parameters(model_na)
    train_loss_nadam /= len(train_loader.dataset)
    train_losses_nadam.append((step, train_loss_nadam))
    print(f'Epoch {epoch}, Step {step}, Loss: {loss.item()}')

    model_na.eval()
    test_loss_nadam = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model_na(images)
            test_loss_nadam += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss_nadam /= len(test_loader.dataset)
    test_losses_nadam.append((step, test_loss_nadam))
    aca = 100. * correct / len(test_loader.dataset)
    print(f' NAdam: Test set: Average loss: {test_loss_nadam}, Train set: {train_loss_nadam} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({aca}%)')
    ac_na.append(aca)
    

fig, ax = plt.subplots(2, 3)
####Plotting them all
train_steps, train_loss_SGD = zip(*train_losses)
test_steps, test_loss_SGD = zip(*test_losses)


train_steps, train_loss_adamm = zip(*train_losses_adam)
test_steps, test_loss_adamm = zip(*test_losses_adam)

train_steps, train_loss_nn = zip(*train_losses_nadam)
test_steps, test_loss_nn = zip(*test_losses_nadam)


#Plotting Training  and validation loss 
ax[0,0].plot(range(1, n_epochs+1), test_loss_nn, label='Test Loss', linestyle='--', color='blue')
ax[0,0].plot(range(1, n_epochs+1), train_loss_nn, label='Training Loss', color='blue')
ax[0,0].set_title('Nadam Loss')
ax[0,0].set_xlabel('Epoch')
ax[0,0].set_ylabel('Loss')
ax[0,0].grid(True)




ax[1,0].plot(range(1, n_epochs+1), ac_na, color='orange')
ax[1,0].set_title('Nadam Accuracy')
ax[1,0].set_xlabel('Epoch')
ax[1,0].set_ylabel('Accuracy')
ax[1,0].grid(True)

# Adam training and validation loss
ax[0,1].plot(range(1, n_epochs+1), test_loss_adamm, label='Test Loss', linestyle='--', color='red')
ax[0,1].plot(range(1, n_epochs+1), train_loss_adamm, label='Training Loss', color='red')
ax[0,1].set_title('Adam Loss')
ax[0,1].set_xlabel('Epoch')
ax[0,1].set_ylabel('Loss')
ax[0,1].grid(True)

# PLotting Adam Accuracy
ax[1,1].plot(range(1, n_epochs+1), acc_, color='darkorange')
ax[1,1].set_title('Adam Accuracy')
ax[1,1].set_xlabel('Epoch')
ax[1,1].set_ylabel('Accuracy')
ax[1,1].grid(True)

# SGD training and validation loss
ax[0,2].plot(range(1, n_epochs+1), train_loss_SGD, label='Test Loss', linestyle='--', color='green')
ax[0,2].plot(range(1, n_epochs+1), test_loss_SGD, label='Training Loss', color='green')
ax[0,2].set_title('SGD Loss')
ax[0,2].set_xlabel('Epoch')
ax[0,2].set_ylabel('Loss')
ax[0,2].grid(True)

# Plotting SGD Accuracy
ax[1,2].plot(range(1, n_epochs+1), acc_sgd, color='darkorange')
ax[1,2].set_title('SGD Accuracy')
ax[1,2].set_xlabel('Epoch')
ax[1,2].set_ylabel('Accuracy')
ax[1,2].grid(True)

plt.tight_layout()
plt.savefig('reg.png')
plt.show()