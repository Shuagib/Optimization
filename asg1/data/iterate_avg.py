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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



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
optimizer = optim.SGD(model.parameters(), lr= 0.3) #Intilize learning rate
swa_model  = optim.swa_utils.AveragedModel(model) #Averaging model weights pr. iteration
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5) #It will change learning rate each time 

train_losses = []
test_losses = []
acc = []
step = 0
decayinglr = [0.32]
for epoch in range(n_epochs):
    model.train()
    train_loss_SGD = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(images)
        loss_SGD = criterion(output, labels)
        train_loss_SGD += criterion(output, labels).item()
        loss_SGD.backward()
        batchlr = optimizer.step()
        step += 1
    scheduler.step()
    swa_model.update_parameters(model)
    train_loss_SGD /= len(train_loader)
    train_losses.append((step, train_loss_SGD))
    decayinglr.append(scheduler.get_lr()[0]) 
    print(f'Epoch {epoch}, Step {step}, Loss: {loss_SGD.item()}')
    print(f'lr { scheduler.get_lr()} lr list {decayinglr}', )
    

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
    print(f'Baseline SGD Test set: Average loss: {test_loss_SGD}, Train set: {train_loss_SGD} \
        Accuracy: {correct}/{len(test_loader.dataset)} ({ac_val}%)')
    acc.append(ac_val)





#  plot train and test losses for SGD Baseline
train_steps, train_loss_SGD = zip(*train_losses)
test_steps, test_loss_SGD = zip(*test_losses)

fig, ax = plt.subplots(1,3)
ax[0].plot(range(1,n_epochs+1), test_loss_SGD, label='Test Loss  SGD', linestyle='--', color = 'blue')
ax[0].plot(range(1,n_epochs+1), train_loss_SGD, label='Training Loss SGD',color='blue')
ax[0].set_title(' Training and validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

ax[1].plot(range(0,n_epochs+1), decayinglr, label = 'SGD ',color = 'green')
ax[1].set_title('Learning rate: ExponentialLR')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Learning rate')
ax[1].grid()

ax[2].plot(range(1,n_epochs +1), acc,color = 'orange')
ax[2].set_title('Accuracy test')
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('Accuracy')
ax[2].grid()

 
plt.grid(True)
#Creating a box that isn't too large
plt.tight_layout()
plt.savefig('Accuracy_iteatemethod.png', bbox_inches='tight')
plt.show()





# Create data loaders.


# Epoch 0, Step 1, Loss: 2.2992544174194336
# full batch size SGD Test set: Average loss: 2.2760017191528514, Train set: 2.2992544174194336         Accuracy: 1144/10000 (11.4399995803833%)
# Epoch 1, Step 2, Loss: 2.2758026123046875
# full batch size SGD Test set: Average loss: 2.2433934561006583, Train set: 2.2758026123046875         Accuracy: 3138/10000 (31.3799991607666%)
# Epoch 2, Step 3, Loss: 2.2430551052093506
# full batch size SGD Test set: Average loss: 2.1894149977690094, Train set: 2.2430551052093506         Accuracy: 3585/10000 (35.849998474121094%)
# Epoch 3, Step 4, Loss: 2.1889073848724365
# full batch size SGD Test set: Average loss: 2.101333569569193, Train set: 2.1889073848724365         Accuracy: 4402/10000 (44.02000045776367%)
# Epoch 4, Step 5, Loss: 2.1006367206573486
# full batch size SGD Test set: Average loss: 1.967674787636775, Train set: 2.1006367206573486         Accuracy: 4718/10000 (47.18000030517578%)
# Epoch 5, Step 6, Loss: 1.9668816328048706
# full batch size SGD Test set: Average loss: 1.8115000550154667, Train set: 1.9668816328048706         Accuracy: 4752/10000 (47.52000045776367%)
# Epoch 6, Step 7, Loss: 1.8107725381851196
# full batch size SGD Test set: Average loss: 1.6900820193017365, Train set: 1.8107725381851196         Accuracy: 4867/10000 (48.66999816894531%)
# Epoch 7, Step 8, Loss: 1.6894973516464233
# full batch size SGD Test set: Average loss: 1.623711497920334, Train set: 1.6894973516464233         Accuracy: 4961/10000 (49.61000061035156%)
# Epoch 8, Step 9, Loss: 1.6232185363769531
# full batch size SGD Test set: Average loss: 1.5959555457352073, Train set: 1.6232185363769531         Accuracy: 4994/10000 (49.939998626708984%)
# Epoch 9, Step 10, Loss: 1.5954980850219727
# full batch size SGD Test set: Average loss: 1.589197133756747, Train set: 1.5954980850219727         Accuracy: 5005/10000 (50.04999923706055%)
# Training loop
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr= 0.9) #Intilize learning rate
# swa_model  = optim.swa_utils.AveragedModel(model) #Averaging model weights pr. iteration
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10) #It will change learning rate each time


# Epoch 0, Step 938, Loss: 0.455269455909729
# Baseline SGD Test set: Average loss: 0.44186134180825226, Train set: 0.5741125271518601         Accuracy: 8385/10000 (83.8499984741211%)
# Epoch 1, Step 1876, Loss: 0.34925949573516846
# Baseline SGD Test set: Average loss: 0.38076501107139954, Train set: 0.3621360699552844         Accuracy: 8583/10000 (85.83000183105469%)
# Epoch 2, Step 2814, Loss: 0.21631628274917603
# Baseline SGD Test set: Average loss: 0.3425143601693166, Train set: 0.31407840213160526         Accuracy: 8728/10000 (87.27999877929688%)
# Epoch 3, Step 3752, Loss: 0.4887586236000061
# Baseline SGD Test set: Average loss: 0.33331062744377526, Train set: 0.2789913438705366         Accuracy: 8777/10000 (87.7699966430664%)
# Epoch 4, Step 4690, Loss: 0.18647176027297974
# Baseline SGD Test set: Average loss: 0.28906063336855287, Train set: 0.2512660244825298         Accuracy: 8947/10000 (89.47000122070312%)
# Epoch 5, Step 5628, Loss: 0.2354133129119873
# Baseline SGD Test set: Average loss: 0.29549975005114915, Train set: 0.22797275744060844         Accuracy: 8931/10000 (89.30999755859375%)
# Epoch 6, Step 6566, Loss: 0.28161054849624634
# Baseline SGD Test set: Average loss: 0.2752798169281832, Train set: 0.2097631217296253         Accuracy: 9001/10000 (90.01000213623047%)
# Epoch 7, Step 7504, Loss: 0.11021572351455688
# Baseline SGD Test set: Average loss: 0.2738002862330455, Train set: 0.19840931143365434         Accuracy: 9024/10000 (90.23999786376953%)
# Epoch 8, Step 8442, Loss: 0.14891542494297028
# Baseline SGD Test set: Average loss: 0.2738002862330455, Train set: 0.19533936651562578         Accuracy: 9024/10000 (90.23999786376953%)
# Epoch 9, Step 9380, Loss: 0.19648505747318268
# Baseline SGD Test set: Average loss: 0.2738891867980076, Train set: 0.196624319625101         Accuracy: 9017/10000 (90.16999816894531%)