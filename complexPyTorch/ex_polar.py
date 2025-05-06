import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from complexPyTorch.polarComplexLayers import PolarConv2d, NaivePolarBatchNorm2d, PolarLinear
from ..polarComplexFunctions import polar_relu, polar_max_pool2d
from ..comPolar64 import ComPolar64

import time


batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class PolarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PolarConv2d(1, 10, 5, 1)
        self.bn = NaivePolarBatchNorm2d(10)
        self.conv2 = PolarConv2d(10, 20, 5, 1)
        self.fc1 = PolarLinear(4*4*20, 500)
        self.fc2 = PolarLinear(500, 10)

    def forward(self, x):
        #x = ComPolar64.from_cartesian(x)
        x = self.conv1(x)
        x = polar_relu(x)
        x = polar_max_pool2d(x, 2, 2)
        x = self.bn(x)
        x = self.conv2(x)
        x = polar_relu(x)
        x = polar_max_pool2d(x, 2, 2)
        x = ComPolar64(
            x.get_magnitude().view(-1, 4*4*20),
            x.get_phase().view(-1, 4*4*20)
        )
        x = self.fc1(x)
        x = polar_relu(x)
        x = self.fc2(x)
        logits = x.get_magnitude()
        logits = F.log_softmax(logits, dim=1)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PolarNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        polar_data = ComPolar64.from_cartesian(data)
        optimizer.zero_grad()
        output = model(polar_data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )


start_time = time.time()
# Run training on 50 epochs

#epoch_times = []
for epoch in range(50):
    epoch_start = time.time()
    train(model, device, train_loader, optimizer, epoch)
    epoch_end = time.time()
    #epoch_times.append(epoch_end - epoch_start)
    print(f"Epoch {epoch} time: {epoch_end - epoch_start:.2f} seconds")

end_time = time.time()  # total time end
torch.save(model.state_dict(), "models/ex_polar_wts.pth")
print(f"Total training time: {end_time - start_time:.2f} seconds")