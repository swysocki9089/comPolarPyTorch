import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from comPolar64 import ComPolar64
import time  # added for timing

batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 10, 5, 1)
        self.bn = ComplexBatchNorm2d(10)
        self.conv2 = ComplexConv2d(10, 20, 5, 1)
        self.fc1 = ComplexLinear(4*4*20, 500)
        self.fc2 = ComplexLinear(500, 10)

    def forward(self, x):
        #x = ComPolar64.from_cartesian(x)  # input assumed cartesian
        x = x.apply_cartesian_function(self.conv1)
        x = x.apply_cartesian_function(complex_relu)
        x = x.apply_cartesian_function(lambda t: complex_max_pool2d(t, 2, 2))
        x = x.apply_cartesian_function(self.bn)
        x = x.apply_cartesian_function(self.conv2)
        x = x.apply_cartesian_function(complex_relu)
        x = x.apply_cartesian_function(lambda t: complex_max_pool2d(t, 2, 2))

        # flatten manually in polar
        mag = x.get_magnitude().view(-1, 4*4*20)
        phase = x.get_phase().view(-1, 4*4*20)
        x = ComPolar64(mag, phase)

        x = x.apply_cartesian_function(self.fc1)
        x = x.apply_cartesian_function(complex_relu)
        x = x.apply_cartesian_function(self.fc2)

        logits = x.get_magnitude()
        logits = F.log_softmax(logits, dim=1)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
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

# Start total timer
total_start_time = time.time()

# Run training for 50 epochs
for epoch in range(50):
    epoch_start_time = time.time()
    train(model, device, train_loader, optimizer, epoch)

    epoch_end_time = time.time()
    print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

# End total timer
total_end_time = time.time()
print(f"Total training time: {total_end_time - total_start_time:.2f} seconds.")

# Save model weights
torch.save(model.state_dict(), 'models/ex_cart_wts.pth')
print("Model weights saved to models/ex_cart_wts.pth")
