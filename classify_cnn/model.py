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
    def __init__(self, p_smallest=0.0, p_block=0.0, p_reflect=0.0, p_filter=0.0):
        super(Net, self).__init__()

        # p_block + p_reflect + p_filter <= 1
        self.p_smallest = p_smallest
        self.p_block = p_block
        self.p_reflect = p_reflect
        self.p_filter = p_filter

        # 2 convolutional layers + 1 fully connected layer (lusch et al)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # max pool after conv layers
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    
    def damage_smallest(self): # connections are blocked to simulate nonuse of connections
        def do_damage_smallest(param, p):
            with torch.no_grad():
                flat = param.view(-1)
                num_to_zero = int(len(flat) * p)

                if num_to_zero == 0:
                    return  # skip for speed

                # get indices of smallest weights
                _, indices = torch.topk(flat.abs(), num_to_zero, largest=False)
                flat[indices] = 0.0

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                do_damage_smallest(module.weight, self.p_smallest)
    
    def damage_fas(self):
        def do_damage_fas(param, p_block, p_reflect, p_filter):
            if p_block + p_reflect + p_filter > 1:
                raise ValueError("percentages for fas damage types must not exceed 100%")

            with torch.no_grad():
                flat = param.view(-1)

                nonzero_indices = (flat!=0).nonzero(as_tuple=True)[0]
                num_nonzero_indices = nonzero_indices.numel()

                # percentage of weights damaged will be taken from the number of nonzero weights
                # simulated fas damage occurs after energy constraint blockage
                num_block = int(num_nonzero_indices * p_block)
                num_reflect = int(num_nonzero_indices * p_reflect)
                num_filter = int(num_nonzero_indices * p_filter)

                shuffled_indices = nonzero_indices[torch.randperm(num_nonzero_indices)]

                indices_block = shuffled_indices[:num_block]
                indices_reflect = shuffled_indices[num_block:num_block+num_reflect]
                indices_filter = shuffled_indices[num_block+num_reflect:num_block+num_reflect+num_filter]

                # low pass filter stuff (lusch et al)
                weights_to_filter = flat[indices_filter]                # get weights before transformation
                signs = weights_to_filter.sign()                        # get signs of weights
                high_weight = torch.quantile(flat.abs(), 0.95)          # get high_weight, should be in the 95th percentile for all weights
                normalized_weights = weights_to_filter / high_weight    # scale weights to mostly between -1 and 1
                x = normalized_weights
                gaussian_noise = np.sqrt(0.05) * np.random.randn(len(x),)
                transformed_weights = -0.2774 * x**2 + 0.9094 * x - 0.0192 + gaussian_noise
                filtered_weights = transformed_weights * signs * high_weight # rescale

                # do damage
                flat[indices_block] = 0.0                  # blockage: set weights to 0
                flat[indices_reflect] *= 0.5               # reflection: halve weights
                flat[indices_filter] = filtered_weights    # filter: low pass filter stuff - see above

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight'):
                    do_damage_fas(module.weight, self.p_block, self.p_reflect, self.p_smallest) # apply damage only to connection weights, and not bias weights




# FIND OPTIMAL SPARSIFICATION PARAMETER FOR ENERGY CONSTRAINT SIMULATION

percentages = np.arange(0, 1.01, 0.01)
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
                for data, target in train_loader:
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

        score = float(correct/len(test_loader.dataset))

        testing_accuracies.append(score)

    max_score = sum(testing_accuracies)/len(testing_accuracies)
    print(max_score)
    accuracies.append(max_score)



# find local maxima to better find our p_smallest from plot
maxima = []
for i in range(1, len(accuracies) - 1):
    if accuracies[i] > accuracies[i-1] and accuracies[i] > accuracies[i+1]: maxima.append((percentages[i], accuracies[i]))
print("MAXIMA!!!!")
print(maxima)

plt.figure(figsize=(8, 5))
plt.plot(percentages, accuracies)
plt.xlabel('percentage of smallest weights damaged')
plt.ylabel('mean testing accuracy')
plt.grid(True)
plt.show()
print("ACCURACIES!!!!")