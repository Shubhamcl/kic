import sys
sys.path.insert(0,'C:\\Users\\sj_x3\\Desktop\\code\\git\\kac_independence_measure\\')

from kac_independence_measure import KacIndependenceMeasure

import numpy as np
import matplotlib.pyplot as plt
import torch

torch.manual_seed(0)
np.random.seed(0)

num_epochs = 10 # For testing

device_cpu = torch.device("cpu")# if torch.cuda.is_available() else "cpu")

fc1s = np.load('./fc1.npy').reshape((num_epochs,-1,600))
fc2s = np.load('./fc2.npy').reshape((num_epochs,-1,120))
fc3s = np.load('./fc3.npy').reshape((num_epochs,-1,10))

# KAC models
k_fc1 = KacIndependenceMeasure(600, 10, lr=0.05,  input_projection_dim = 0, weight_decay=0.01)
k_fc2 = KacIndependenceMeasure(120, 10, lr=0.05,  input_projection_dim = 0, weight_decay=0.01)

for epoch_num in range(num_epochs):#[0,5,9]:
    history_dep1, history_dep2 = [], []

    fc1_data = fc1s[epoch_num]
    fc2_data = fc2s[epoch_num]
    fc3_data = fc3s[epoch_num]

    for batch in range(0,10000, 100):
        fc1 = fc1_data[batch: batch+100]
        fc2 = fc2_data[batch: batch+100]
        fc3 = fc3_data[batch: batch+100]

        fc1, fc2, fc3 = torch.Tensor(fc1).to(device_cpu), torch.Tensor(fc2).to(device_cpu), torch.Tensor(fc3).to(device_cpu)

        dep1 = k_fc1(fc1, fc3)
        dep2 = k_fc2(fc2, fc3)

        history_dep1.append(dep1.detach().numpy())
        history_dep2.append(dep2.detach().numpy())

    if epoch_num in [0,1,2,9]:
        # plt.plot(history_dep1, label=f"fc1_epoch{epoch_num}")
        plt.plot(history_dep2, label=f"fc2_epoch{epoch_num}")

plt.legend(loc="upper right")
plt.ylim(0,1)
# plt.title('FC1 vs FC3')
plt.title('FC2 vs FC3')
plt.show()
