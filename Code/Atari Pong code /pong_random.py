# -*- coding: utf-8 -*-
"""

@author: billy

"""
from pong_skeleton_libraries import skeleton_helper_functions
from pong_skeleton_libraries import pong_env_wrappers as env_wrappers
from pong_skeleton_libraries import skeleton_dqn_architectures

import numpy as np
import time # just to have timestampsAgent in the file
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


agent_type = 'RANDOM' # this changes across files
print(torch.__version__)

"""
ENVIRONMENT AND DIFFICULTY SETTINGS
"""
ENV_NAME = "PongNoFrameskip-v4"
difficulty = 1    # 1, 2, 4 for boxing

print ('Difficulty SELECTED!!!!!!  ', difficulty)


env = env_wrappers.make_train_env(ENV_NAME, difficulty, save_every_x_games = 1_000, name_of_game = ENV_NAME, 
                                  net_type = agent_type, stacked_frames = 2)

# What game is being played?
print(env.env.game)
# What are the available difficulties?
print(env.ale.getAvailableDifficulties())



"""
Setting the training parameters.
"""
# Train on CUDA IF POSSIBLE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on: ' + str(device))


#################################

MAXIMUM_STEPS_ALLOWED = 500_000


##########################################
#####################################
####################################

# SELECT THE DESIRED VERSION OF DQN 
net = skeleton_dqn_architectures.NATURE_CNN_DQN(env.observation_space.shape,env.action_space.n).to(device) 
target_net = skeleton_dqn_architectures.NATURE_CNN_DQN(env.observation_space.shape,env.action_space.n).to(device)

# PERFORMANCE TRACKER IN TENSORBOARD 
writer = SummaryWriter(log_dir = "runs/Paper runs/" + agent_type +" -Diff " + str(difficulty) + " " + ENV_NAME)



state = env.reset()
episode_reward = 0.0
episode_rewards = []
steps_count = 0


buffer = None

while True:
      
  steps_count += 1
  
  epsilon = 1.0
  action = skeleton_helper_functions.select_action(epsilon,state,env,None,None)
  
  state, reward, is_done, _ =  env.step(action)

  episode_reward += reward

  
  if is_done: # if the episode is done, then log the episode reward

      episode_rewards.append(episode_reward) # holds reward for each game
      mean_reward_100 = np.mean(episode_rewards[-100:])
      
      skeleton_helper_functions.print_and_save_logs(writer, steps_count, epsilon, 
                                                    mean_reward_100, 
                                                    episode_reward,
                                                    len(episode_rewards))
      
      episode_reward = 0.0 
      state  = env.reset()
      

  if steps_count == MAXIMUM_STEPS_ALLOWED:
        print("Maximum steps allowed reached, training finished.")
        break 






















































