import gym
import matplotlib.pyplot as plt 
import cv2, torch
import torch.nn as nn
import numpy as np

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
        for e_i in range(self._args.episodes):
            ob = self._env.reset()
            ob = np.hstack((ob, [np.log(1)])) 
            done = False
            total_reward = 0
            self._env.render()
            cur_frame = self.current_frame()            
            # state = cur_frame.repeat(1,self._args.n_frames,1,1)
            state = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).cuda()
            while not done:
                # Get current action
                action, _ = self._actor(state)
                
                cur_frame = self.current_frame()            
                # Perform action in environment
                ob, reward, done, _ = self._env.step(action)
                total_reward += reward
                #print("ob", np.hstack(ob, [np.log(total_reward)])) 
                ob = np.hstack((ob, [np.log(total_reward + 1)])) 
                

                # Add memory step
                if done:
                    next_state = None
                    print(reward, action)
                    reward = -reward
                else:
                    next_state = torch.tensor(ob, dtype=torch.float32).unsqueeze(0).cuda()
                
                if next_state is None:
                    e_t = Experience(state.clone(), action, reward, next_state)
                else:
                    e_t = Experience(state.clone(), action, reward, next_state.clone())
                self._actor.add_ex(e_t)
                state = next_state
                
                #if e_i > 0 and self._actor.replay_len() > self._args.batch_size:
                #    self._actor.train()

            print("Episode", e_i, "total reward", total_reward)
            self._env.reset()
            self._actor.save()
            for i in range(4):
                self._actor.train()

        self._env.close()



