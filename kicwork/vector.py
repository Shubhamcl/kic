import sys
sys.path.insert(0,'C:\\Users\\sj_x3\\Desktop\\code\\git\\kac_independence_measure\\')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from model_def import FashionCNN, FashionDataset
from kac_independence_measure import KacIndependenceMeasure

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")# if torch.cuda.is_available() else "cpu")

# Function to make hooks
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Load model
model = torch.load('./models/model.t7')
model.to(device)

# KAC models
k_fc1 = KacIndependenceMeasure(600, 10, lr=0.05,  input_projection_dim = 0, weight_decay=0.01)
k_fc2 = KacIndependenceMeasure(120, 10, lr=0.05,  input_projection_dim = 0, weight_decay=0.01)

# Attaching hooks
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))
model.fc3.register_forward_hook(get_activation('fc3'))

# Data loading
test_csv = pd.read_csv("./data/fashion-mnist_test.csv")
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=100)

history_dep1 = []
history_dep2 = []
for _ in range(4):

    for images, labels in test_loader:
        
        # Generate CNN outputs
        images, labels = images.to(device), labels.to(device)
        test = Variable(images.view(100, 1, 28, 28))
        outputs = model(test)

        # Pass activations through KAC models
        fc1 = torch.clone(activation['fc1']).to(device_cpu)
        fc2 = torch.clone(activation['fc2']).to(device_cpu)
        fc3 = torch.clone(activation['fc3']).to(device_cpu)

        dep1 = k_fc1(fc1, fc3)
        dep2 = k_fc2(fc2, fc3)

        history_dep1.append(dep1.detach().numpy())
        history_dep2.append(dep2.detach().numpy())

plt.plot(history_dep1, label="fc1")
plt.plot(history_dep2, label="fc2")
plt.legend(loc="upper right")
plt.title('FC1 and FC2 vs FC3')
# plt.show()
plt.savefig('./layer_dependence.png')
sys.exit()
# Loop through test_loader
# Get output
# Run through kac measure models

sample = next(iter(test_loader))

image, label = sample[0], sample[1]
image = image.to(device)
image = Variable(image.view(100, 1, 28, 28))

pred = model(image)

# print(pred)
# print(label)

# x = torch.randn(1, 25)
# output = model(x)
print(type(activation['fc1']))
print(activation['fc1'].shape)
print(activation['fc2'].shape)
print(activation['fc3'].shape)


# 