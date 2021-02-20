import gym
import matplotlib.pyplot as plt 
import cv2, torch
import torch.nn as nn

from dqn import DQNActor
from replay_memory import Experience

DO_NOTHING_ACTION = 1


class Trainer:
    def __init__(self, args, actor, env):
        self._args = args
        self._env = env
        self._actor = actor
        
    def current_frame(self):
        """Get current frame."""
        # Get frame
        frame = self._env.render(mode='rgb_array')

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize the frame
        frame = cv2.resize(frame, (self._args.img_dim, self._args.img_dim), interpolation = cv2.INTER_LINEAR)
        
        # Normalize frame to [-1, 1]
        frame = frame / 127.5 - 1
        
        return torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)

    def run(self):
        self._env.reset()
        for e_i in range(self._args.episodes):
            done = False
            total_reward = 0
            cur_frame = self.current_frame()            
            state = cur_frame.repeat(1,self._args.n_frames,1,1)

            while not done:
                # Get current action
                action, _ = self._actor(state)
                
                # Perform action in environment
                _, reward, done, _ = self._env.step(action)
                
                # Get next state
                next_state = state
                next_state = next_state.roll(-1, 1)
                next_state[:, self._args.n_frames - 1] = self.current_frame()

                # Add memory step
                if done:
                    next_state = None
                    print(reward, action)
                total_reward += reward
                e_t = Experience(state, action, reward, next_state)
                self._actor.add_ex(e_t)
                state = next_state
                
                if e_i > 0 and self._actor.replay_len() > self._args.batch_size:
                    self._actor.train()

            print("Episode", e_i, "total reward", total_reward)
            self._env.reset()
            self._actor.save()

        self._env.close()



