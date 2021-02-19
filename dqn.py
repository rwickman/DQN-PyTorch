import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the actor seperate from the model

class DQNActor():
    def __init__(self):
        self._dqnet = DQNetwork()
        self._epsilon = 0.05
    def __call__(self, x):
        return self._dqnet(x)
    
    def get_action(self, q_values):
        if self._epsilon >= np.random.rand():
            # Perform random action
            action = np.random.randint(len(q_values))
        else:
            # Perfrom action that maximizes expected return
            action =  np.argmax(q_values)
        return action, q_values[action]


class DQNetwork(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        # 4 input frames, 16 filters of size 8x8
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        # 32 filter of size 4x4
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.flatten = nn.Flatten()
        # 256 units 
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_dim)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        # Outputs state-action value q(s,a) for every action
        return self.fc2(x)
    
    # def action(self, x):
    #     if np.random.rand() > 