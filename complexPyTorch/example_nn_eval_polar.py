import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from polarComplexLayers import PolarConv2d, NaivePolarBatchNorm2d, PolarLinear
from polarComplexFunctions import polar_relu, polar_max_pool2d
from comPolar64 import ComPolar64


class PolarNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PolarConv2d(1, 10, 5, 1)
        self.bn = NaivePolarBatchNorm2d(10)
        self.conv2 = PolarConv2d(10, 20, 5, 1)
        self.fc1 = PolarLinear(4*4*20, 500)
        self.fc2 = PolarLinear(500, 10)

    def forward(self, x):
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


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            polar_data = ComPolar64.from_cartesian(data)

            output = model(polar_data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PolarNet().to(device)
    model.load_state_dict(torch.load("polar_model.pt", map_location=device))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    evaluate(model, device, test_loader)


if __name__ == "__main__":
    main()
