import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json, math, os

from replay_memory import ReplayMemory
from replay_memory import PrioritizedExpReplay

# Make the actor seperate from the model

EPS_START = 0.9
EPS_END = 0.05

class DQNActor:
    def __init__(self, args, action_dim):
        self._args = args
        self._action_dim = action_dim
        self._dqnet = DQNetwork(args, self._action_dim).cuda()
        self._dqnet_target = DQNetwork(args, self._action_dim).cuda()
        self._replay_memory = PrioritizedExpReplay(self._args)#ReplayMemory(args)
        self._optimizer = optim.RMSprop(self._dqnet.parameters(), lr=self._args.lr)
        self._epsilon = args.epsilon
        self._loss_fn = nn.SmoothL1Loss()
        self._num_steps = 0
        if self._args.load_model:
            self.load()
      

    def __call__(self, x):
        return self.get_action(self._dqnet(x))
    
    def get_action(self, q_values, argmax=False):
        q_values = torch.squeeze(q_values)
        if not argmax and self.epsilon_threshold() >= np.random.rand():
            # Perform random action
            action = np.random.randint(self._action_dim)
        else:
            with torch.no_grad():
                # Perfrom action that maximizes expected return
                action =  q_values.max(0)[1]
        action = int(action)
        return action, q_values[action]
    
    def train(self):
        """Train Q-Network over batch of sampled experience."""
        # Get sample of experience
        is_ws, exs, indices = self._replay_memory.sample(self._args.batch_size)
        #exs = self._replay_memory.sample()
    
        td_targets = torch.zeros(self._args.batch_size).cuda()
        states = torch.zeros(self._args.batch_size, 5).cuda()
        next_states = torch.zeros(self._args.batch_size, 5).cuda()
        rewards = torch.zeros(self._args.batch_size).cuda()
        next_state_mask = torch.zeros(self._args.batch_size).cuda()
        actions = []
        # Create state-action values
        for i, e_t in enumerate(exs):
            states[i] = e_t.state
            actions.append(e_t.action)
            rewards[i] = e_t.reward
            if e_t.next_state is not None:
                next_states[i] = e_t.next_state 
                next_state_mask[i] = 1

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64).cuda()
        #print("self._dqnet(states)", self._dqnet(states))
        #print("q_values",self._dqnet(states))
        q_values = self._dqnet(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next  = self._dqnet(next_states)
        #print("q_next", q_next)
        q_next_target  = self._dqnet_target(next_states).detach()
        for i in range(self._args.batch_size):
            if next_state_mask[i] == 0:
                td_targets[i] = rewards[i]
            else:
                # Get the argmax next action for DQN
                action, _ = self.get_action(q_next[i], True)
                
                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                td_targets[i] = rewards[i] + self._args.gamma * q_next_target[i, action]


        # Train model
        self._optimizer.zero_grad()
        td_errors = q_values - td_targets

        #loss = self._loss_fn(q_values, td_targets)
        loss = torch.mean(td_errors ** 2  *  is_ws)
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self._dqnet.parameters(), 2.0)
        # print("self._dqnet.fc1.weight.grad.max()", self._dqnet.fc1.weight.grad.max())
        # print("self._dqnet.fc2.weight.grad.max()", self._dqnet.fc2.weight.grad.max())

        self._optimizer.step()
        self._num_steps += 1
        self._replay_memory.update_priorities(indices, td_errors.detach())

        # Update target policy
        if (self._num_steps + 1) % self._args.target_update_step == 0:
            print("loss", loss)
            print("epsilon_threshold", self.epsilon_threshold())
            print("q_values", q_values)
            print("td_targets", td_targets)
            print("td_errors", td_errors)
            print("replay_len", self.replay_len())
            self._dqnet_target.load_state_dict(self._dqnet.state_dict())
        
    def add_ex(self, e_t):
        """Add a step of experience."""
        with torch.no_grad():
            _, q_value = self.get_action(self._dqnet(e_t.state))
            if e_t.next_state is not None:
                next_action, _ = self.get_action(self._dqnet(e_t.next_state))
                q_next_target = self._dqnet_target(e_t.next_state)[0, next_action]
                td_target = e_t.reward + self._args.gamma *  q_next_target
            else:
                td_target = e_t.reward

            td_error = td_target - q_value

        self._replay_memory.add(e_t, td_error)


    def replay_len(self):
        return self._replay_memory.current_capacity()
    
    def epsilon_threshold(self):
        return self._args.min_epsilon + (self._args.epsilon - self._args.min_epsilon) * \
            math.exp(-1. * self._num_steps / self._args.epsilon_decay)

    def save(self):
        model_dict = {
            "DQN" : self._dqnet.state_dict(),
            "optimizer" : self._optimizer.state_dict()
        }
        
        torch.save(model_dict, self._args.model)
        with open("model_meta.json", "w") as f:
            json.dump({"num_steps" : self._num_steps}, f)

    def load(self):
        model_dict = torch.load(self._args.model)
        self._dqnet.load_state_dict(model_dict["DQN"])
        self._dqnet_target.load_state_dict(model_dict["DQN"])
        self._optimizer.load_state_dict(model_dict["optimizer"])
        if os.path.exists("model_meta.json"):
            with open("model_meta.json") as f:
                d = json.load(f)
                self._num_steps = d["num_steps"]

# class DQNetwork(nn.Module):
#     def __init__(self, args, action_dim=2):
#         super().__init__()
#         # 4 input frames, 16 filters of size 8x8
#         self.conv1 = nn.Conv2d(args.n_frames, 16, 8, stride=4)
#         # 32 filter of size 4x4
#         self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
#         self.flatten = nn.Flatten()
        
        
#         conv2d_out_dim = self._conv2d_out(
#             self._conv2d_out(args.img_dim, 8, 4), 4, 2)        
#         num_units = int(conv2d_out_dim ** 2 * 32)
#         print("conv2d_out_dim", conv2d_out_dim)
#         print("num_units", num_units)
#         self.fc1 = nn.Linear(num_units, 256)
#         self.fc2 = nn.Linear(256, action_dim)

    
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         # Outputs state-action value q(s,a) for every action
#         return self.fc2(x)
    
#     def _conv2d_out(self, dim, kernel_size, stride):
#         # W - F + 2P / S
#         return (dim - kernel_size) / stride + 1


class DQNetwork(nn.Module):
    def __init__(self, args, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Outputs state-action value q(s,a) for every action
        return self.fc3(x)

class Actor(nn.Module):
    def __init__(self, args, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.out = nn.Softmax()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Outputs state-action value q(s,a) for every action
        return self.out(self.fc3(x))

