# -*- coding: utf-8 -*-
"""
@author: billy

Helper fucntion for carrying out training...
"""

import torch
import torch.nn as nn
import numpy as np
import collections
from tensorboardX import SummaryWriter

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward','done', 'new_state'])


def take_env_step(net, state, env,  epsilon, buffer, device = 'cuda'):
      
    action = select_action(epsilon, state, env, device, net)

    # do step in the environment, also return some info
    new_state, reward, is_done, _ =  env.step(action)

    #create new Experience transition
    exp = Experience(state, action, reward, is_done, new_state)
      
    # Add experience to the Buffer
    buffer.append(exp)


    return reward, new_state, is_done
        



def select_action(e,state,env,device,main_net):
      action=None
      with torch.no_grad():
            
            if np.random.random() >= e:
                  
                  state = np.array([state], copy=False)
                  state_tensor = torch.tensor(state).to(device) #create pytorch tensor for state
                  action_v = main_net(state_tensor).max(1)[1].view(1, 1)
                  action = int(action_v.item())
                  
            else:
                 
                  action = env.action_space.sample() #random action to take 
      
      return action             


def print_and_save_logs(writer, steps_count, epsilon, mean_reward_100, reward, episodes_completed):
    
    writer.add_scalar("epsilon", epsilon, steps_count)
    writer.add_scalar("episode reward", reward, steps_count)
    writer.add_scalar("mean_reward_100", mean_reward_100, steps_count)

    print("Steps taken: %d, Episodes completed: %d , Current epsilon: %f, Mean reward achieved: %f " % (steps_count,episodes_completed, epsilon, mean_reward_100))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity



    def sample(self, batch_size):
        
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states, actions, rewards, dones, next_states = zip(*[self.memory[index] for index in indices])

        return np.array(states,dtype=np.float32), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)



    def to_pytorch_tensors(self, batch, device):

        states, actions, rewards, dones, next_states = batch
        tensor_states = torch.tensor(states).to(device)
        tensor_next_states = torch.tensor(next_states).to(device)
        tensor_actions = torch.tensor(actions, dtype=torch.long).to(device) #actions must be of type Long for some reason
        tensor_rewards = torch.tensor(rewards).to(device)
        done_flags = torch.BoolTensor(dones).to(device)

        
        non_final_mask = ~ done_flags #not operation on the boolean tensor
        non_final_next_states = tensor_next_states[non_final_mask] #tensor of non final states, to pass through the target net
    
        return tensor_states,non_final_next_states,tensor_actions,tensor_rewards,non_final_mask


    def __len__(self):
        return len(self.memory)


    


