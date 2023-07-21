import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 40, 3, padding=1)
        self.conv4 = nn.Conv2d(40, 10, 3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP layer

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.global_avg_pool(x)  # Global Average Pooling
        x = x.view(-1, 10)  # Modify this line for the desired output size
        return F.log_softmax(x, dim=1)  # Specify dim=1 for the log_softmax operation
    
    def train_model(self, device, train_loader, optimizer, num_epochs):
        self.to(device)
        self.train()
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                pbar.set_description(desc=f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

    def test_model(self, device, test_loader):
        self.to(device)
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))