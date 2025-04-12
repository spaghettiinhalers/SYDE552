import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

mnist = torchvision.datasets.MNIST(root='.', download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist, np.arange(5000)), 
                                           batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist, np.arange(5000, 10000)), 
                                          batch_size=1000, shuffle=True)

class Net(nn.Module):
    def __init__(self, p_smallest=0.0, p_random=0.0):
        super(Net, self).__init__()

        self.p_smallest = p_smallest
        self.p_random = p_random
        # 2 convolutional layers + 1 fully connected layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # set the size of the convolution to 5x5, and have 12 of them
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # make sure to do max pooling after the convolution layers
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    
    def damage_smallest(self):
        def do_damage_smallest(param, p):
            with torch.no_grad():
                # Flatten weights and compute absolute magnitude
                flat = param.view(-1)
                num_to_zero = int(len(flat) * p)

                if num_to_zero == 0:
                    return  # Skip if p is too small

                # Get indices of the smallest weights
                _, idx = torch.topk(flat.abs(), num_to_zero, largest=False)

                # Zero them out
                flat[idx] = 0.0

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                do_damage_smallest(module.weight, self.p_smallest)
    
    def damage_random(self):
        def do_damage_random(param, p):
            with torch.no_grad():
                mask = (torch.rand_like(param) > p).float()
                param.mul_(mask)

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    do_damage_random(module.weight, self.p_random) # apply damage only to connection weights, and not bias weights



# FIND OPTIMAL SPARSIFICATION PARAMETER FOR ENERGY CONSTRAINT SIMULATION

percentages = np.arange(0, 1.001, 0.001)
accuracies = []

for percentage in percentages:
    print(percentage)
    testing_accuracies = []
    
    for i in range(10): # train, damage, and test model ten times to get max
        network = Net(p_smallest = percentage)

        # create the learning rule
        optimizer = torch.optim.SGD(network.parameters(), 
                                    lr=0.1,   # learning rate
                                    momentum=0.5)

        for j in range(10):
            network.train()      # configure the network for training
            for k in range(10):  # train the network 10 times
                correct = 0
                for data, target in train_loader:       # working in batchs of 1000
                    optimizer.zero_grad()               # initialize the learning system
                    output = network(data)              # feed in the data 
                    loss = F.nll_loss(output, target)   # compute how wrong the output is
                    loss.backward()                     # change the weights to reduce error
                    optimizer.step()                    # update the learning rule
                    
                    pred = output.data.max(1, keepdim=True)[1]           # compute which output is largest
                    correct += pred.eq(target.data.view_as(pred)).sum()  # compute the number of correct outputs

        # damage smallest weights
        network.damage_smallest()

        correct = 0
        network.eval()
        for data, target in test_loader:    # go through the test data once (in groups of 1000)
            output = network(data)                               # feed in the data
            pred = output.data.max(1, keepdim=True)[1]           # compute which output is largest
            correct += pred.eq(target.data.view_as(pred)).sum()  # compute the number of correct outputs
        # update the list of testing accuracy values
        score = float(correct/len(test_loader.dataset))

        testing_accuracies.append(score)

    max_score = sum(testing_accuracies)/len(testing_accuracies)
    print(max_score)
    accuracies.append(max_score)



# find local maxima to better find our p_smallest from plot
maxima = []
for i in range(1, len(accuracies) - 1):
    if accuracies[i] > accuracies[i-1] and accuracies[i] > accuracies[i+1]: maxima.append((percentages[i], accuracies[i]))
print(maxima)

plt.figure(figsize=(8, 5))
plt.plot(percentages, accuracies)
plt.xlabel('percentage of smallest weights damaged')
plt.ylabel('mean testing accuracy')
plt.grid(True)
plt.show()