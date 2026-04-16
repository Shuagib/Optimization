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
optimizer = optim.SGD(model.parameters(), lr= 0.9)


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


## SGD
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
        Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
  





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


# Epoch 0, Step 12000, Loss: 0.18764866888523102
# Mini SGD Test set: Average loss: 0.4333452764590075, Train set: 0.6592818779537144         Accuracy: 8429/10000 (84.29000091552734%)
# Epoch 1, Step 24000, Loss: 0.7720698714256287
# Mini SGD Test set: Average loss: 0.38226827884175973, Train set: 0.3883578435268913         Accuracy: 8601/10000 (86.01000213623047%)
# Epoch 2, Step 36000, Loss: 0.5037024021148682
# Mini SGD Test set: Average loss: 0.3630763161334263, Train set: 0.3390231965318186         Accuracy: 8674/10000 (86.73999786376953%)
# Epoch 3, Step 48000, Loss: 0.02052803710103035
# Mini SGD Test set: Average loss: 0.3298624603991296, Train set: 0.31097554669257566         Accuracy: 8815/10000 (88.1500015258789%)
# Epoch 4, Step 60000, Loss: 0.18592920899391174
# Mini SGD Test set: Average loss: 0.3183241964905125, Train set: 0.2914770603387345         Accuracy: 8835/10000 (88.3499984741211%)
# Epoch 5, Step 72000, Loss: 0.33072593808174133
# Mini SGD Test set: Average loss: 0.3076916441890844, Train set: 0.27521272502440114         Accuracy: 8905/10000 (89.05000305175781%)
# Epoch 6, Step 84000, Loss: 0.44478192925453186
# Mini SGD Test set: Average loss: 0.29677916104626506, Train set: 0.2624731833848273         Accuracy: 8922/10000 (89.22000122070312%)
# Epoch 7, Step 96000, Loss: 0.9116008877754211
# Mini SGD Test set: Average loss: 0.30449887127823133, Train set: 0.2508949779036011         Accuracy: 8890/10000 (88.9000015258789%)
# Epoch 8, Step 108000, Loss: 0.08856727182865143
# Mini SGD Test set: Average loss: 0.2885714533507444, Train set: 0.24027986404110682         Accuracy: 8980/10000 (89.80000305175781%)
# Epoch 9, Step 120000, Loss: 0.021943364292383194
# Mini SGD Test set: Average loss: 0.28829763478534237, Train set: 0.23117498861937322         Accuracy: 8975/10000 (89.75%)
# number of training samples: 60000
# number of testing samples: 10000
# datatype of the 1st training sample:  torch.FloatTensor
# size of the 1st training sample:  torch.Size([1, 28, 28])
# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
# Shape of y: torch.Size([64]) torch.int64
# +--------------+------------+
# |   Modules    | Parameters |
# +--------------+------------+
# | conv1.weight |    150     |
# |  conv1.bias  |     6      |
# | conv2.weight |    2400    |
# |  conv2.bias  |     16     |
# |  fc1.weight  |   30720    |
# |   fc1.bias   |    120     |
# |  fc2.weight  |   10080    |
# |   fc2.bias   |     84     |
# |  fc3.weight  |    840     |
# |   fc3.bias   |     10     |
# +--------------+------------+
# Total Trainable Params: 44426
# 30720
# 10080
# Epoch 0, Step 1, Loss: 2.3046462535858154
# full batch size SGD Test set: Average loss: 2.2898607223656526, Train set: 2.3046462535858154         Accuracy: 1000/10000 (10.0%)
# Epoch 1, Step 2, Loss: 2.2895519733428955
# full batch size SGD Test set: Average loss: 2.2732411858382497, Train set: 2.2895519733428955         Accuracy: 1824/10000 (18.239999771118164%)
# Epoch 2, Step 3, Loss: 2.2727773189544678
# full batch size SGD Test set: Average loss: 2.2425775421652823, Train set: 2.2727773189544678         Accuracy: 2985/10000 (29.850000381469727%)
# Epoch 3, Step 4, Loss: 2.241830825805664
# full batch size SGD Test set: Average loss: 2.179340976059057, Train set: 2.241830825805664         Accuracy: 3321/10000 (33.209999084472656%)
# Epoch 4, Step 5, Loss: 2.1781065464019775
# full batch size SGD Test set: Average loss: 2.03864710346149, Train set: 2.1781065464019775         Accuracy: 4487/10000 (44.869998931884766%)
# Epoch 5, Step 6, Loss: 2.036863327026367
# full batch size SGD Test set: Average loss: 1.7564407670573823, Train set: 2.036863327026367         Accuracy: 4682/10000 (46.81999969482422%)
# Epoch 6, Step 7, Loss: 1.7546576261520386
# full batch size SGD Test set: Average loss: 1.8138067243964808, Train set: 1.7546576261520386         Accuracy: 3139/10000 (31.389999389648438%)
# Epoch 7, Step 8, Loss: 1.80947744846344
# full batch size SGD Test set: Average loss: 2.5322526093501194, Train set: 1.80947744846344         Accuracy: 945/10000 (9.449999809265137%)
# Epoch 8, Step 9, Loss: 2.5304126739501953
# full batch size SGD Test set: Average loss: 2.394351989600309, Train set: 2.5304126739501953         Accuracy: 1768/10000 (17.68000030517578%)
# Epoch 9, Step 10, Loss: 2.3963096141815186
# full batch size SGD Test set: Average loss: 2.11471440457994, Train set: 2.3963096141815186         Accuracy: 1604/10000 (16.040000915527344%)
# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
# Shape of y: torch.Size([64]) torch.int64
# +--------------+------------+
# |   Modules    | Parameters |
# +--------------+------------+
# | conv1.weight |    150     |
# |  conv1.bias  |     6      |
# | conv2.weight |    2400    |
# |  conv2.bias  |     16     |
# |  fc1.weight  |   30720    |
# |   fc1.bias   |    120     |
# |  fc2.weight  |   10080    |
# |   fc2.bias   |     84     |
# |  fc3.weight  |    840     |
# |   fc3.bias   |     10     |
# +--------------+------------+
# Total Trainable Params: 44426
# 30720
# 10080
# Epoch 0, Step 938, Loss: 0.7700589895248413
# Baseline SGD Test set: Average loss: 0.5581217858062428, Train set: 0.7361105777847488         Accuracy: 7838/157 (78.37999725341797%)
# Epoch 1, Step 1876, Loss: 0.48054784536361694
# Baseline SGD Test set: Average loss: 0.5164653434874905, Train set: 0.43128219543934376         Accuracy: 7967/157 (79.66999816894531%)
# Epoch 2, Step 2814, Loss: 0.3495858609676361
# Baseline SGD Test set: Average loss: 0.3773086298802856, Train set: 0.3686265293627914         Accuracy: 8616/157 (86.16000366210938%)
# Epoch 3, Step 3752, Loss: 0.09479450434446335
# Baseline SGD Test set: Average loss: 0.34974200017512985, Train set: 0.3354985183363022         Accuracy: 8731/157 (87.30999755859375%)
# Epoch 4, Step 4690, Loss: 0.5822596549987793
# Baseline SGD Test set: Average loss: 0.35362001418308087, Train set: 0.3130413601790537         Accuracy: 8702/157 (87.0199966430664%)
# Epoch 5, Step 5628, Loss: 0.1348230093717575
# Baseline SGD Test set: Average loss: 0.32878074908901933, Train set: 0.29516055106099987         Accuracy: 8777/157 (87.7699966430664%)
# Epoch 6, Step 6566, Loss: 0.5123618841171265
# Baseline SGD Test set: Average loss: 0.3187098038519264, Train set: 0.281045419773631         Accuracy: 8851/157 (88.51000213623047%)
# Epoch 7, Step 7504, Loss: 0.22133305668830872
# Baseline SGD Test set: Average loss: 0.32485921078237, Train set: 0.27044916211732667         Accuracy: 8818/157 (88.18000030517578%)
# Epoch 8, Step 8442, Loss: 0.1285751760005951
# Baseline SGD Test set: Average loss: 0.3124188157688281, Train set: 0.26017047404480387         Accuracy: 8848/157 (88.4800033569336%)
# Epoch 9, Step 9380, Loss: 0.24161110818386078
# Baseline SGD Test set: Average loss: 0.30183710803271857, Train set: 0.25069897447917255         Accuracy: 8869/157 (88.69000244140625%)