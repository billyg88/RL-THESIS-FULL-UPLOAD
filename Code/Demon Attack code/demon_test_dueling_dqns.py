# -*- coding: utf-8 -*-
"""
@author: billy
"""
from demon_skeleton_libraries import skeleton_helper_functions
from demon_skeleton_libraries import demon_env_wrappers as env_wrappers
from demon_skeleton_libraries import skeleton_dqn_architectures

import numpy as np
import time 
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import time


device = 'cuda' #run the testing experiments on cpu


env_name  = 'DemonAttackNoFrameskip-v4'
difficulty = 0 # Variable and easy to change!!!! {0-> 'Easy', 1->'Hard'}
agent = '.dat'  # variable and easy to change 


#DuelingDoubleDQNs
#DemonAttackNoFrameskip-v4_diff_0-best-reward=155.45
#DemonAttackNoFrameskip-v4_diff_1-best-reward=78.8
#DemonAttackNoFrameskip-v4_diff_Curriculum_V2-best-reward=184.1


# Must keep an eye on the difficulty setting and the number of frames stacked (Must be the same number as in the training procedure)
save_10_episodes = 10

#Τhis below must be modified
env = env_wrappers.make_test_env(env_name, difficulty, save_10_episodes, name_of_game ='Demon', net_type = 'DQN_'+agent, stacked_frames=2)
env = env.unwrapped
print('Max steps',env._max_episode_steps)

# Make sure to load the correct Agent to test!!!!
net = skeleton_dqn_architectures.Dueling_CNN_DQN(env.observation_space.shape,env.action_space.n).to(device)
state = torch.load('models/DuelingDoubleDQN/'+agent, map_location=lambda stg, _: stg)   #Source directory also changes for each agent type
net.load_state_dict(state)


writer = SummaryWriter(log_dir = "runs/Paper runs/Test runs/" +'DuelingDoubleDQN_ '+ agent +" -Diff " + str(difficulty))
# Agent makes only non-random moves!!! so the epsilon rate is set to 0!
num_of_eps = 100
episode_reward = 0.0 
episode_rewards = []


e = 0.05
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
                  state_tensor = torch.tensor(state).to(device) #create pytorch tensor for state
      
                  action_v = net(state_tensor).max(1)[1].view(1, 1) #get action with the highest value 
                  action_index = int(action_v.item())
                  
            else:
                
                  action_index = env.action_space.sample() #random action to take       
            
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





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    