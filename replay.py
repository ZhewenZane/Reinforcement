import numpy as np


class ReplayBuffer(object):
    def __init__(self,max_size,state_dim,action_dim):
        self.mem_cntr = 0 
        self.mem_size = max_size 
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Create Initial Batch Numpy Array
        self.state_batch = np.zeros((self.mem_size,self.state_dim))
        self.action_batch = np.zeros((self.mem_size,self.action_dim))
        self.next_state_batch = np.zeros((self.mem_size,self.state_dim))
        self.reward_batch = np.zeros((self.mem_size,1))
        self.done_batch = np.zeros((self.mem_size,1),dtype=bool)
        

    def push(self,state,action,next_state,reward,done):
        index = self.mem_cntr % self.mem_size
        self.state_batch[index] = state
        self.action_batch[index] = action
        self.next_state_batch[index] = next_state 
        self.reward_batch[index] = reward
        self.done_batch[index] = done

        self.mem_cntr += 1 
        if self.mem_cntr == self.mem_size:
            print(f"I have reached to my MAX !!!")
    
    def sample(self,batch_size):
        sample_range = min(self.mem_cntr,self.mem_size)
        indexes = np.random.choice(sample_range,batch_size,replace=False)
        states = self.state_batch[indexes]
        actions = self.action_batch[indexes]
        next_states = self.next_state_batch[indexes]
        rewards = self.reward_batch[indexes]
        dones = self.done_batch[indexes]
        
        return states,actions,next_states,rewards,dones

    
    def __len__(self):
        return self.mem_cntr 
    


class PeReplay(object):
    def __init__(self,max_size,state_dim,action_dim,alpha=0.7):
        self.mem_cntr = 0 
        self.mem_size = max_size 
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha 
        # Initialize Maximal D
        self.max_D = 1
        # Create Initial Batch Numpy Array
        self.state_batch = np.zeros((self.mem_size,self.state_dim))
        self.action_batch = np.zeros((self.mem_size,self.action_dim))
        self.next_state_batch = np.zeros((self.mem_size,self.state_dim))
        self.reward_batch = np.zeros((self.mem_size,1))
        self.done_batch = np.zeros((self.mem_size,1),dtype=bool)
        self.priority = np.zeros((self.mem_size,1))
        

    def push(self,state,action,next_state,reward,done):
        index = self.mem_cntr % self.mem_size
        self.state_batch[index] = state
        self.action_batch[index] = action
        self.next_state_batch[index] = next_state 
        self.reward_batch[index] = reward
        self.done_batch[index] = done
        self.priority[index] = self.max_D
        self.mem_cntr += 1 
    
    def sample(self,batch_size):
        sample_range = min(self.mem_cntr,self.mem_size)
        probs = np.power(self.priority.squeeze()[:sample_range],self.alpha)/(np.power(self.priority.squeeze()[:sample_range],self.alpha).sum())
        indexes = np.random.choice(sample_range,batch_size,replace=False,p=probs) # shape as (length,)
        states = self.state_batch[indexes]
        actions = self.action_batch[indexes]
        next_states = self.next_state_batch[indexes]
        rewards = self.reward_batch[indexes]
        dones = self.done_batch[indexes]
        priorities = self.priority[indexes]  # shape as [batch_size,1]
        
        return states,actions,next_states,rewards,dones,indexes,priorities

    def update_priority(self,indexes,priorities):
        # Update the priority according to |TD error|
        for i in range(len(indexes)):
            index = indexes[i]
            self.priority[index] = priorities[i]
            
        # Update the max D value for next-pushes 
        max_priori = priorities.max()
        if max_priori > self.max_D:
            self.max_D = max_priori
        
    
    def __len__(self):
        return self.mem_cntr 
    
