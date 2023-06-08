import gymnasium as gym 
import numpy as np



from agent import DDPG, DDPGPer
from utils import * 
import wandb 

# wandb.login()

# wandb.init(
#     project="DDPG",
#     config={
#         "datasets":"Mujoco-Ant",
#         "epoch:":5000,
#     }
# )


device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('Ant-v4',render_mode=None)

agent = DDPG(env,device)
# agent = DDPGPer(env,device)
noise = OUNoise(env.action_space)
reward_scaling = 1
batch_size = 64

print(f"Device now: {device}")
for episode in range(5000):
    score = 0 
    state = env.reset()[0]
    # print(f"Episode {episode}")
    step = 0
    while True:
        action = agent.get_action(state) 
        action = noise.get_action(action)
        next_state,reward,done,_,_ = env.step(action)
        agent.memory.push(state,action,next_state,reward*reward_scaling,done)
        
        if len(agent.memory) >= batch_size:
            agent.update(batch_size)

        state = next_state
        score += reward 
        
        if done:
            break
    # print(f"The length of the Memory after episode {episode} is {len(agent.memory)}")
    print(f"of episode: {episode}, Average Rewards: {score}")
    # wandb.log({"Average Rewards":score})



