import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)

#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


class QNetwork(nn.Module):
    def __init__(self, state_size,action_size,seed):
        super(QNetwork, self).__init__()        
        h = state_size[0]
        w = state_size[1]
        in_channels = state_size[2]
        
        h1 = int((h-8)/4) + 1
        w1 = int((w-8)/4) + 1
        h2 = int((h1-4)/2) + 1
        w2 = int((w1-4)/2) + 1
        h_out = h2-3+1
        w_out = w2-3+1


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features = h_out*w_out*64,out_features=512)
        self.fc2 = nn.Linear(in_features = 512,out_features=action_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        # print("!!",np.shape(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
