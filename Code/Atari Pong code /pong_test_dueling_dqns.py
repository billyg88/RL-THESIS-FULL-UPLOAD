# -*- coding: utf-8 -*-
"""
@author: billy
"""
from pong_skeleton_libraries import skeleton_helper_functions
from pong_skeleton_libraries import pong_env_wrappers as env_wrappers
from pong_skeleton_libraries import skeleton_dqn_architectures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import time


FPS = 800  
device = 'cuda' 


env_name  = 'PongNoFrameskip-v4'
difficulty = 1 # Variable and easy to change!!!! {1-> 'Easy', 0->'Hard'}
agent = 'PongNoFrameskip-v4_diff_curriculum-best_2.dat'  # variable and easy to change 


#DuelingDoubleDQNs
#PongNoFrameskip-v4_diff_0-best_18
#PongNoFrameskip-v4_diff_1-best_17
#PongNoFrameskip-v4_diff_curriculum-best_2


save_10_episodes = 10
env = env_wrappers.make_test_env(env_name, difficulty, save_10_episodes, name_of_game ='PONG', net_type = 'DQN_'+agent, stacked_frames=2)
net = skeleton_dqn_architectures.Dueling_CNN_DQN(env.observation_space.shape,env.action_space.n).to(device)
state = torch.load('models/DuelingDoubleDQN/'+agent, map_location=lambda stg, _: stg)   #Source directory also changes for each agent type
net.load_state_dict(state)


writer = SummaryWriter(log_dir = "runs/Paper runs/Test runs/" +'DuelingDoubleDQN_ '+ agent +" -Diff " + str(difficulty))
num_of_eps = 50
episode_reward = 0.0 # Total reward for an episode
episode_rewards = []


e = 0.02
steps_count = 0
episode_steps = 0

for i in range(num_of_eps):          
    print('Episode :', i)
    state = env.reset()  
    
    while (True): 
            episode_steps += 1
            steps_count +=1
          
            #Uncomment to render environment
            #env.render()
            
            if np.random.random() >= e:
                  state = np.array([state], copy=False)
                  state_tensor = torch.tensor(state).to(device) 
                  action_v = net(state_tensor).max(1)[1].view(1, 1) 
                  action_index = int(action_v.item())
                  
            else:
                
                  action_index = env.action_space.sample() 
                  
            
            state, step_reward, done, info = env.step(action_index)
            episode_reward += step_reward
            
            if done:
                print('Steps taken: ',episode_steps) 
                episode_rewards.append(episode_reward)
                #LOG PERFORMANCE with tensorboard 
                writer.add_scalar("Rewards", episode_reward, len(episode_rewards)-1)
                writer.add_scalar("Mean Reward", np.mean(episode_rewards[-10:]),steps_count)
                writer.add_scalar('Cummmulative reward',np.array(episode_rewards).sum(),len(episode_rewards)-1)
                writer.add_scalar('Steps per episode',episode_steps, len(episode_rewards))
                
                print(episode_reward)
                print('\n')
                
                episode_reward = 0.0
                episode_steps = 0
                break
              
        
print(episode_rewards)
print('Cummulative reward: ',np.array(episode_rewards).sum())





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    