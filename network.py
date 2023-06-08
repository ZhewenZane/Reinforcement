import torch.nn as nn
import torch 
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self,input_size,hid_size,output_size):
        super(Critic,self).__init__()
        self.linear1 = nn.Linear(input_size,hid_size)
        self.linear2 = nn.Linear(hid_size,hid_size)
        self.linear3 = nn.Linear(hid_size,output_size)
        self.network_init()

    def forward(self,state,action):
        x = torch.cat((state,action),-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
    
        return x 
    
    
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
  

class Actor(nn.Module):
    def __init__(self,input_size,hid_size,output_size):
        super(Actor,self).__init__()
        self.linear1 = nn.Linear(input_size,hid_size)
        self.linear2 = nn.Linear(hid_size,hid_size)
        self.linear3 = nn.Linear(hid_size,output_size)
        self.network_init()
        
    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        
        return x 
    
    
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
    
    
