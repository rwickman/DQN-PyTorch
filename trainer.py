import gym
import matplotlib.pyplot as plt 
import cv2, torch
import torch.nn as nn

from dqn import DQNActor
from replay_memory import Experience

DO_NOTHING_ACTION = 1


class Trainer:
    def __init__(self, args, actor):
        self._args = args
        self._actor = actor
        self._env = gym.make('MountainCar-v0')
        
    
    def current_frame(self):
        # Get current frame
        frame = self._env.render(mode='rgb_array')
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize the frame
        frame = cv2.resize(frame, (84,84), interpolation = cv2.INTER_LINEAR)
        
        return torch.from_numpy(frame)


    def run(self):
        self._env.reset()
        for e_i in range(self._args.episodes):
            done = False
            first_state = True
            while not done:
                state = torch.zeros(1, 2, 84, 84)
                # Add current frame to state
                #state[:, 0] = current_frame()
                cur_reward = 0
                for i in range(2):
                    #print(frame.shape)
                    if first_state:
                        # Don't do anything
                        self._env.step(DO_NOTHING_ACTION)
                    else:
                        _, reward, done, _ = self._env.step(action)
                        cur_reward += reward
                    state[:, i] = self.current_frame()
                    #print(frame.shape)
                    
                    #plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
                    #plt.show()
                
                action, q_value = self._actor(state)
                
                if first_state:
                    first_state = False
                else:   
                    # Add memory step
                    if done:
                        state = None
                    e_t = Experience(prev_state, action, reward, state)
                    self._actor.add_ex(e_t)

                if e_i > 0 and self._actor.replay_len() > self._args.batch_size:
                    #print(self._actor.replay_len())
                    self._actor.train()

                prev_state = state
            print("Episode", e_i)
            self._env.reset()
            self._actor.save()

        self._env.close()



