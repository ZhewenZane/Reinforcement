import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network import *
from replay import ReplayBuffer, PeReplay
from utils import convert_to_tensor, OUNoise  


class DDPG(object):
    def __init__(self,env,device,hid_size=256,actor_learning_rate=3e-4,critic_learning_rate=3e-4,gamma=0.99,tau=0.005,max_memory_size=1000000):
        self.gamma=gamma 
        self.tau=tau
        self.action_size = env.action_space.shape[0]
        self.state_size = env.observation_space.shape[0]
        self.max_memory_size = max_memory_size
        self.device = device 
        # self.noise = OUNoise(self.action_size,666)
        # Networks 
        self.actor = Actor(self.state_size,hid_size,self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size,hid_size,self.action_size).to(self.device)
        self.critic = Critic(self.state_size+self.action_size,hid_size,1).to(self.device)
        self.critic_target = Critic(self.state_size+self.action_size,hid_size,1).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_param.data.copy_(param.data)

        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        # Replay Buffer 
        self.memory = ReplayBuffer(self.max_memory_size,self.state_size, self.action_size) 

    def get_action(self,state):
        state = torch.from_numpy(state).to(torch.float32).to(self.device)
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy()
        # noise = self.noise.sample()
        # action = np.clip(action + noise,-1,1)

        return action

    def update(self,batch_size):
        states,actions,next_states,rewards,dones = self.memory.sample(batch_size)
        states,actions,next_states,rewards,dones = convert_to_tensor(states,actions,next_states,rewards,dones)
        # rewards = torch.unsqueeze(rewards,1) # This is for making the rewards' shape the same as next_Q below 
        # print(f"States Shape:{states.shape}, Action Shape: {actions.shape}, Next_States's shape: {next_states.shape}, Rewards Shape:{rewards.shape}")
        
        # Transfer to device
        states = states.to(self.device)
        actions  = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)


        # Critic Loss 
        Qvals = self.critic.forward(states,actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states,next_actions)
        Qprime = rewards + (~dones)*self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals,Qprime.detach())

        # Actor Loss 
        policy_loss = -self.critic.forward(states,self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_params, params in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1-self.tau))

        for target_params, params in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1-self.tau))
        
        # not completed tasks maybe for plots




class DDPGPer(object):
    def __init__(self,env,device,hid_size=256,actor_learning_rate=3e-4,critic_learning_rate=1e-3,gamma=0.99,tau=1e-2,max_memory_size=1000000,alpha=0.7,beta=0.5,epsilon=1e-6):
        self.gamma=gamma 
        self.tau=tau
        self.action_size = env.action_space.shape[0]
        self.state_size = env.observation_space.shape[0]
        self.max_memory_size = max_memory_size
        self.device = device 
        self.alpha = alpha
        self.beta = beta 
        self.epsilon = epsilon 
        # Networks 
        self.actor = Actor(self.state_size,hid_size,self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size,hid_size,self.action_size).to(self.device)
        self.critic = Critic(self.state_size+self.action_size,hid_size,1).to(self.device)
        self.critic_target = Critic(self.state_size+self.action_size,hid_size,1).to(self.device)

        for target_param, param in zip(self.actor.parameters(),self.actor_target.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic.parameters(),self.critic_target.parameters()):
            target_param.data.copy_(param.data)

        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        # For now create Replay Buffer with PER  
        self.memory = PeReplay(self.max_memory_size,self.state_size, self.action_size,alpha=self.alpha) 

    def get_action(self,state):
        state = torch.from_numpy(state).to(torch.float32).to(self.device)
        action = self.actor.forward(state)
        action = action.cpu().detach().numpy()

        return action

    def update(self,batch_size):
        states,actions,next_states,rewards,dones,indexes,priorities = self.memory.sample(batch_size) # priorities used for weights calculations
        states,actions,next_states,rewards,dones = convert_to_tensor(states,actions,next_states,rewards,dones)
        # rewards = torch.unsqueeze(rewards,1) # This is for making the rewards' shape the same as next_Q below 
        # print(f"States Shape:{states.shape}, Action Shape: {actions.shape}, Next_States's shape: {next_states.shape}, Rewards Shape:{rewards.shape}")
        
        # Transfer to device
        states = states.to(self.device)
        actions  = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones =dones.to(self.device)

        # Critic Loss 
        Qvals = self.critic.forward(states,actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states,next_actions.detach())
        Qprime = rewards + (~dones)*self.gamma * next_Q
        # Calculate weights bias and update critic loss
        priorities_tensor = torch.from_numpy(priorities).to(self.device)
        weights = torch.pow(batch_size*priorities_tensor,-self.beta*0.5)
        weights = weights/weights.max()

        critic_loss = self.critic_criterion(weights*Qvals,weights*Qprime)

        # Update memory batch priority 
        Q1 = Qvals.clone().detach()
        Q2 = Qprime.clone().detach()
        td = torch.abs(Q1-Q2) 
        eps = torch.ones_like(td) * self.epsilon
        td += eps 
        priorities_new = td.cpu().squeeze().numpy()
        self.memory.update_priority(indexes=indexes,priorities=priorities_new)

        # Actor Loss 
        policy_loss = -self.critic.forward(states,self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_params, params in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1-self.tau))

        for target_params, params in zip(self.critic_target.parameters(),self.critic.parameters()):
            target_params.data.copy_(params.data * self.tau + target_params.data * (1-self.tau))
        
        # not completed tasks maybe for plots