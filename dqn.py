import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json, math, os

from replay_memory import ReplayMemory

# Make the actor seperate from the model

EPS_START = 0.9
EPS_END = 0.05

class DQNActor:
    def __init__(self, args):
        self._args = args
        self._dqnet = DQNetwork()
        self._dqnet_target = DQNetwork()
        self._replay_memory = ReplayMemory(args)
        self._optimizer = optim.RMSprop(self._dqnet.parameters(), lr=args.lr)
        self._epsilon = args.epsilon
        self._loss_fn = nn.MSELoss()
        self._num_steps = 0
        if self._args.load_model:
            self.load()
      

    def __call__(self, x):
        return self.get_action(self._dqnet(x))
    
    def get_action(self, q_values, argmax=False):
        
        if argmax or self.epsilon_threshold() >= np.random.rand():
            # Perform random action
            action = torch.tensor([np.random.randint(q_values.shape[1], size=q_values.shape[0])])
        else:
            with torch.no_grad():
                # Perfrom action that maximizes expected return
                action =  q_values.max(1)[1].detach().view(-1, 1)
        action = int(action[0,0])
        return action, q_values[0, action]
    
    def train(self):
        """Train Q-Network over batch of sampled experience."""
        # Get sample of experience
        exs = self._replay_memory.sample()
        q_values = torch.zeros(self._args.batch_size)#torch.Tensor([self._dqnet(e_t.state)[e_t.action] for e_t in exps])
        td_targets = torch.zeros(self._args.batch_size)
        # Create state-action values
        for i, e_t in enumerate(exs):
            q_values[i] = self._dqnet(e_t.state)[0, e_t.action]
            if e_t.next_state is not None:
                _, next_q_value = self.get_action(
                    self._dqnet_target(e_t.state),
                    True)
                td_targets[i] = e_t.reward + self._args.gamma * next_q_value
        
        # Train model
        loss = self._loss_fn(q_values, td_targets)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._num_steps += 1
        
        # Update target policy
        if (self._num_steps + 1) % self._args.target_update_step == 0:
            
            print("epsilon_threshold", self.epsilon_threshold())
            print("q_values", q_values)
            print("td_targets", td_targets)
            print("replay_len", self.replay_len())
            self._dqnet_target.load_state_dict(self._dqnet.state_dict())
            #target_net.load_state_dict(policy_net.state_dict())

    def copy_to_target(self):
        """Copy to parameters to target network."""
        pass
        
    def add_ex(self, e_t):
        """Add a step of experience."""
        self._replay_memory.append(e_t)

    def replay_len(self):
        return self._replay_memory.current_capacity()
    
    def epsilon_threshold(self):
        f_1 = self._args.min_epsilon + (self._args.epsilon - self._args.min_epsilon)
        f_2 =  math.exp(-1. * self._num_steps / self._args.epsilon_decay)
        return f_1 * f_2

    def save(self):
        torch.save(self._dqnet.state_dict(), self._args.model)
        with open("model_meta.json", "w") as f:
            json.dump({"num_steps" : self._num_steps}, f)

    def load(self):
        self._dqnet.load_state_dict(torch.load(self._args.model))
        self._dqnet_target.load_state_dict(torch.load(self._args.model))
        if os.path.exists("model_meta.json"):
            with open("model_meta.json") as f:
                d = json.load(f)
                self._num_steps = d["num_steps"]

class DQNetwork(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        # 4 input frames, 16 filters of size 8x8
        self.conv1 = nn.Conv2d(2, 16, 8, stride=4)
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