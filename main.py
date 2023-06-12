import gymnasium as gym 
import numpy as np



from agent import DDPG, DDPGPer
from utils import * 
import wandb 

'''
    wandb.login()

    wandb.init(
        project="DDPG",
        config={
            "datasets":"Mujoco-Ant",
            "epoch:":2000,
        }
    )
'''


device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('Ant-v4',render_mode=None)

agent = DDPG(env,device)
# agent = DDPGPer(env,device)


noise = OUNoise(env.action_space)
batch_size = 32
learning_start_size = 1000

print(f"Device now: {device}")
for episode in range(3000): 
    score = 0 
    state = env.reset()[0]
    # step = 0
    # done = False
    # print(f"Episode {episode}")
    for step in range(6001):
    # while not done:
        action= noise.get_action(agent.get_action(state),t=step)
        next_state,reward,done,truncate,_ = env.step(action)
        agent.memory.push(state,action,next_state,reward,done)
        
        if len(agent.memory) >= learning_start_size:
            agent.update(batch_size)

        state = next_state
        score += reward 
        # step += 1 
        if done: 
            break
        
    # print(f"The length of the Memory after episode {episode} is {len(agent.memory)}")
    print(f"of episode: {episode}, Average Returns: {score}, Step number: {step}")
    # wandb.log({"Average Training Rewards":score})}
    

