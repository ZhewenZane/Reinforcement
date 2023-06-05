import torch
import numpy as np
from torch.nn import functional as F
import gymnasium as gym
env = gym.make('Ant-v4',render_mode=None)
state = env.reset()[0]
print(f"Initial State:{state}")
for i in range(10000):
    # env.render()
    action = env.action_space.sample()
    print(f"Action: {action}")
    env.step(action)
env.close()