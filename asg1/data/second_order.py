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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #Train on
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


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

model = LeNet5_SGD().to(device)

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
criterion = nn.CrossEntropyLoss()
SGD = optim.SGD(model.parameters(), lr= 0.5) #Intilize learning rate
optimizer  = optim.LBFGS(model.parameters(),lr = 0.01,max_iter=10,history_size= 10,line_search_fn= 'strong_wolfe') #Averaging model weights pr. iteration
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= 10) #It will change learning rate each time 






train_losses = []
test_losses = []
acc = []
step = 0
decayinglr = [0.32]
for epoch in range(n_epochs):
    model.train()
    train_loss_LBGS = 0
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            return loss 
        optimizer.step(closure)
        step += 1
        output2 = model(images)
        loss = closure()
        running_loss += loss.item()
        train_loss_LBGS += criterion(output2, labels).item()
    scheduler.step()
    train_loss_LBGS /= len(train_loader)
    train_losses.append((step, train_loss_LBGS))
    decayinglr.append(scheduler.get_lr()[0]) 
    print(f'Epoch {epoch}, Step {step}, Loss: {train_loss_LBGS }')
    print(f'lr { scheduler.get_lr()} lr list {decayinglr}')
    print(f'running loss {running_loss}')
    

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
    ac_val = 100. * correct / len(test_loader.dataset)
    print(f'Baseline SGD Test set: Average loss: {test_loss_SGD}, Train set: {train_loss_LBGS} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({ac_val}%)')
    acc.append(ac_val)








train_steps, train_loss_SGD = zip(*train_losses)
test_steps, test_loss_SGD = zip(*test_losses)

fig, ax = plt.subplots(1,3)

ax[0].plot(range(1,n_epochs+1), test_loss_SGD, label='Test Loss  SGD', linestyle='--', color = 'blue')
ax[0].plot(range(1,n_epochs+1), train_loss_SGD, label='Training Loss SGD',color='blue')
ax[0].set_title(' Training and validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

ax[1].plot(range(0,n_epochs+1), decayinglr, label = 'SGD ',color = 'green')
ax[1].set_title('Learning rate: CosineAnnealing')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Learning rate')
ax[1].grid()

ax[2].plot(range(1,n_epochs +1), acc,color = 'Orange')
ax[2].set_title('Accuracy test')
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('Accuracy')
ax[2].grid()


plt.grid(True)
#Creating a box that isn't too large
plt.tight_layout()
plt.savefig('Accuracy_iteatemethod.png', bbox_inches='tight')
plt.show()


