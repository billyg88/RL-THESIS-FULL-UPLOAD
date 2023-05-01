# -*- coding: utf-8 -*-
"""
@author: billy
"""

import numpy as np
from gym import wrappers
from gym import envs
#print(envs.registry.all())
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


"""
Architecture used in the Nature paper: "Human-level control through deep reinforcement learning".
"""

class  NATURE_CNN_DQN(nn.Module):

    def __init__(self, input_dimensions, n_actions):
        # super constructor?
        super(NATURE_CNN_DQN, self). __init__()

        self.fully_connected_input = 7*7*64 #self.get_convolution_output(input_dimensions)

        self.conv_trans1 = nn.Conv2d(input_dimensions[0], 32, kernel_size = 8, stride = 4)

        self.conv_trans2  = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        
        self.conv_trans3 =  nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(self.fully_connected_input, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):

        x = x.float()

        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))

        # flatten vector for the sequential part of the NN 
        x = x.view(x.size()[0], -1)

        result = self.fully_connected_layers(x)

        return result   



"""
Dueling DQN architecture as introduced in 
"Dueling Network Architectures for Deep Reinforcement Learning"
"""

class Dueling_CNN_DQN(nn.Module):

    def __init__(self,input_dimensions,n_actions):

        super(Dueling_CNN_DQN, self).__init__()

        self.fully_connected_input = 7*7*64 

        self.conv_trans1 = nn.Conv2d(input_dimensions[0], 32, kernel_size = 8, stride = 4)

        self.conv_trans2  = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        
        self.conv_trans3 =  nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.state_value = nn.Sequential(
            # FCNN PART
            nn.Linear(self.fully_connected_input, 256),
            nn.ReLU(),
            nn.Linear(256, 1) #Value of the state
        )

        self.action_advantages = nn.Sequential(
            # FCNN PART
            nn.Linear(self.fully_connected_input, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions) #Value of the actions
        )




    def forward(self, x):
        #define path through the network.
        x = x.float()

        # First perfrom feature extraction
        conv_features = F.relu(self.conv_trans1(x))
        conv_features = F.relu(self.conv_trans2(conv_features))
        conv_features = F.relu(self.conv_trans3(conv_features))

        conv_features = conv_features.view(conv_features.size(0),-1)

        state_value = self.state_value(conv_features)
        actions_advantages = self.action_advantages(conv_features)

        result = state_value + actions_advantages

        normalized_result = result - actions_advantages.mean(dim = 1, keepdim = True)  # substract the mean or max of these advantages

        return  normalized_result





#OPTIMIZE THE MEAN SQUARED ERROR
criterion = nn.MSELoss()    
    
# Naive version of the Q-loss, by lopping through each sample and collecting the loss
# And summming it into the batch loss, which is then back-propagated

def naive_dqn_loss_V2(batch, main_net, target_net, gamma, batch_size = 32, device = "cuda", optimizer = 'opt'):

    tensor_states,tensor_next_non_final_states,tensor_actions,tensor_rewards,non_final_mask = batch 

    # create predicted (NN output)
    predicted_state_action_values = main_net(tensor_states) # dims = batch_size,number_of_actions

    next_state_values = torch.zeros((batch_size,predicted_state_action_values.size()[1]),  device=device) 
    
    #print(next_state_values)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(tensor_next_non_final_states) #
        next_state_values = next_state_values.detach() 

    next_state_values = next_state_values * gamma
    
    batch_loss = 0.0

    for prediction, reward, action,  tensor in zip(predicted_state_action_values,tensor_rewards,tensor_actions, next_state_values):

        target = tensor.max() + reward 

        target = target.detach() # detach, no gradient flow to the ground Truth!!!
        prediction = prediction[action]

        sample_loss = criterion(prediction,target)
        
        batch_loss += sample_loss

    return batch_loss / batch_size





#  Calculate the loss of the objective function (MSE), using all the parameters such as Gamma  
def simple_dqn_loss(batch, main_net, target_net, gamma, batch_size = 32, device="cuda"):
      
    tensor_states,tensor_next_non_final_states,tensor_actions,tensor_rewards,non_final_mask = batch   

    predicted_state_action_values = main_net(tensor_states).gather(1, tensor_actions.unsqueeze(-1)).squeeze(-1)
    
    next_state_values = torch.zeros(batch_size, device=device) 

    with torch.no_grad():
        #compute next state values using Target network
        next_state_values[non_final_mask] = target_net(tensor_next_non_final_states).max(1)[0] #only take the values and not the indices
        #detach from pytorch computation graph 
        next_state_values = next_state_values.detach() 
        

    #target values using Bellman aproximation
    target_state_action_values = tensor_rewards + (gamma * next_state_values ) 
                                   
    
    return criterion(predicted_state_action_values,target_state_action_values)        
    
    



"""
To be used with the Dueling DQN arhicteecture so that the implementation 
fits the description of the "Dueling Double DQN "

"""
def double_dqn_loss(batch, main_net, target_net, gamma, batch_size = 32 , device="cuda"):

      tensor_states,tensor_next_non_final_states,tensor_actions,tensor_rewards,non_final_mask = batch
      
      #predicted q values unsing the main network
      predicted_state_action_values = main_net(tensor_states).gather(1, tensor_actions.unsqueeze(-1)) #predicted q values using the main network
      predicted_state_action_values = predicted_state_action_values.squeeze(-1)

      next_state_values = torch.zeros(batch_size, device=device) 
      
      with torch.no_grad():

            next_state_acts = main_net(tensor_next_non_final_states).max(1)[1].view(tensor_next_non_final_states.shape[0], 1)#using the main network, to get the next state action

            next_state_values[non_final_mask] = target_net(tensor_next_non_final_states).gather(1, next_state_acts).squeeze(-1) #calculate the q-value for the next state action to be taken using the target network
            next_state_values = next_state_values.detach()
      
      #target values using Bellman aproximation
      target_state_action_values =  tensor_rewards + (gamma * next_state_values) #target values for our loss function

    
      return criterion(predicted_state_action_values,target_state_action_values)     #Simple MSE loss between target and predicted

