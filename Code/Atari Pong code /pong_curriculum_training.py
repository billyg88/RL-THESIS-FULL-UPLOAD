# -*- coding: utf-8 -*-
"""
@author: billy

The purpose of this script is to create a  well -  performing RL agent, and save the weights, this agent should be trained on curriculum learning
By first being trained on easy difficulty =1, and then harder difficulty=0   (SETTINGS FOR PONG)

"""

from pong_skeleton_libraries import skeleton_helper_functions
from pong_skeleton_libraries import pong_env_wrappers as env_wrappers 
from pong_skeleton_libraries import skeleton_dqn_architectures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter


RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


agent_type = 'DuelingDoubleDQN'
print(torch.__version__)

"""
ENVIRONMENT AND DIFFICULTY SETTINGS
"""
ENV_NAME = "PongNoFrameskip-v4"


env_easy  = env_wrappers.make_train_env_curriculum(ENV_NAME, 1, save_every_x_games = 10, 
                                                   name_of_game = ENV_NAME, net_type = agent_type,
                                                   stacked_frames = 2)

env_hard  = env_wrappers.make_train_env_curriculum(ENV_NAME, 0, save_every_x_games = 10, 
                                                   name_of_game = ENV_NAME, net_type = agent_type,
                                                   stacked_frames = 2)

env_easy_2  = env_wrappers.make_train_env_curriculum(ENV_NAME, 1, save_every_x_games = 10, 
                                                   name_of_game = ENV_NAME, net_type = agent_type,
                                                   stacked_frames = 2)

# What game is being played?
print(env_easy.env.game)
# What are the available difficulties?
print(env_easy.ale.getAvailableDifficulties())


"""
Setting the training parameters.
"""
# Train on CUDA IF POSSIBLE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on: ' + str(device))

##############################
##########################
####################
NETS_SYNCH_TARGET = 1_000  
REPLAY_BUFFER_START_SIZE_FOR_LEARNING = 10_000  
LEARNING_RATE = 1e-4  
REPLAY_BUFFER_SIZE = 100_000   

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
GAMMA_FACTOR = 0.99
BATCH_SIZE = 32

# Must be expnaded to exploret the 'Hard' environment as well.
EPSILON_DECAY_LAST_FRAME = 300_000 # must allow for exploration in both 'easy' and 'hard' settings.   

#Must add up to 500k 
EASY_STEPS = 200_000 
HARDER_STEPS = 200_000 
FINAL_EASY_STEPS = 100_000

##########################################
#####################################
print(env_easy.observation_space.shape)


if agent_type == "DQN" or agent_type == "DoubleDQN" :
      
      net = skeleton_dqn_architectures.NATURE_CNN_DQN(env_easy.observation_space.shape,env_easy.action_space.n).to(device) 
      target_net = skeleton_dqn_architectures.NATURE_CNN_DQN(env_easy.observation_space.shape,env_easy.action_space.n).to(device)

elif agent_type == "DuelingDoubleDQN" :
      
      net = skeleton_dqn_architectures.Dueling_CNN_DQN(env_easy.observation_space.shape,env_easy.action_space.n).to(device) 
      target_net = skeleton_dqn_architectures.Dueling_CNN_DQN(env_easy.observation_space.shape,env_easy.action_space.n).to(device)


# PERFORMANCE TRACKER IN TENSORBOARD 
writer = SummaryWriter(log_dir = "runs/Paper runs/" + agent_type +" -Diff " + "Curriculum_V2" + " " + ENV_NAME)
print("DQN ARCHITECTURE \n", net)


buffer = skeleton_helper_functions.ReplayMemory(REPLAY_BUFFER_SIZE)
epsilon = EPSILON_START
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


'''
TRAINING LOOP !!!!
            
V2
1) 200K STEPS ON EASY 
2) 200K STEPS ON HARD 
3) 100K STEPS ON EASY    
'''


episode_reward = 0.0
episode_rewards = []
steps_count = 0
max_mean_reward = None



state = env_easy.reset()
for i  in range(EASY_STEPS):
      
  steps_count += 1
  epsilon = max(EPSILON_FINAL, EPSILON_START - steps_count / EPSILON_DECAY_LAST_FRAME)
  step_reward, state, done = skeleton_helper_functions.take_env_step(net, state, env_easy,  epsilon, buffer, device = device) 
  episode_reward += step_reward

  if done: # if the episode is done, then log the episode reward

      episode_rewards.append(episode_reward) # holds reward for each game
      mean_reward_100 = np.mean(episode_rewards[-100:])
      
      
      skeleton_helper_functions.print_and_save_logs(writer, steps_count, epsilon, 
                                                    mean_reward_100, 
                                                    episode_reward,
                                                    len(episode_rewards))
      
      if max_mean_reward is None or max_mean_reward < mean_reward_100:
          torch.save(net.state_dict(), 'models/'+agent_type+'/'+ENV_NAME+'_diff_'+'curriculumV2'+"-best_%.0f.dat" % mean_reward_100)
          
          if max_mean_reward is not None:
              print("Best reward updated %.3f -> %.3f" % (max_mean_reward, mean_reward_100))
          max_mean_reward = mean_reward_100


      # the episode is finished so reset the env and the episode reward
      episode_reward = 0.0
      state = env_easy.reset()
      
      
      
  """
  Load the buffer first!!!!! before synchning or back-propagating and learning on data
  """
  
  if len(buffer) < REPLAY_BUFFER_START_SIZE_FOR_LEARNING:
      print("Populating buffer")
      continue


  if steps_count % NETS_SYNCH_TARGET == 0:
      print('Nets synched')
      target_net.load_state_dict(net.state_dict())



  optimizer.zero_grad()  
  batch = buffer.sample(BATCH_SIZE)  
  batch = buffer.to_pytorch_tensors(batch,device)  
  batch_loss = None
  
  if agent_type == "DQN":
        
        batch_loss = skeleton_dqn_architectures.simple_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
             
  elif agent_type == "DoubleDQN" or agent_type == "DuelingDoubleDQN" : 
        
        batch_loss = skeleton_dqn_architectures.double_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
  batch_loss.backward()
  
  optimizer.step()



      
print('HARD TRAINING STARTED')
episode_reward = 0.0 # episode reward reset to 0

state = env_hard.reset()
for i  in range(HARDER_STEPS):
      
  steps_count += 1
  
  epsilon = max(EPSILON_FINAL, EPSILON_START - steps_count / EPSILON_DECAY_LAST_FRAME)
  step_reward, state, done = skeleton_helper_functions.take_env_step(net, state, env_hard,  epsilon, buffer, device = device) 
  episode_reward += step_reward

  
  if done: # if the episode is done, then log the episode reward

      episode_rewards.append(episode_reward) # holds reward for each game
      mean_reward_100 = np.mean(episode_rewards[-100:])
      
      
      skeleton_helper_functions.print_and_save_logs(writer, steps_count, epsilon, 
                                                    mean_reward_100, 
                                                    episode_reward,
                                                    len(episode_rewards))
      
      if max_mean_reward is None or max_mean_reward < mean_reward_100:
          torch.save(net.state_dict(), 'models/'+agent_type+'/'+ENV_NAME+'_diff_'+'curriculum'+"-best_%.0f.dat" % mean_reward_100)
          
          if max_mean_reward is not None:
              print("Best reward updated %.3f -> %.3f" % (max_mean_reward, mean_reward_100))
          max_mean_reward = mean_reward_100
          
      
      # the episode is finished so reset the env and the episode reward
      episode_reward = 0.0
      state = env_hard.reset()
      
      
      
  """
  Load the buffer first!!!!! before synchning or back-propagating and learning on data
  """
  
  if len(buffer) < REPLAY_BUFFER_START_SIZE_FOR_LEARNING:
      print("Populating buffer")
      continue


  if steps_count % NETS_SYNCH_TARGET == 0:
      print('Nets synched')
      target_net.load_state_dict(net.state_dict())



  optimizer.zero_grad()
  
  batch = buffer.sample(BATCH_SIZE)
  
  batch = buffer.to_pytorch_tensors(batch,device)
  
  
  batch_loss = None

  
  if agent_type == "DQN":
        
        batch_loss = skeleton_dqn_architectures.simple_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
        
        
  elif agent_type == "DoubleDQN" or agent_type == "DuelingDoubleDQN" : 
        
        batch_loss = skeleton_dqn_architectures.double_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
  batch_loss.backward()
  
  optimizer.step()


     
print('FINAL EASY TRAINING STARTED')
episode_reward = 0.0 # episode reward reset to 0

state = env_easy_2.reset()

for i  in range(FINAL_EASY_STEPS):
      
  steps_count += 1
  
  epsilon = max(EPSILON_FINAL, EPSILON_START - steps_count / EPSILON_DECAY_LAST_FRAME)
  step_reward, state, done = skeleton_helper_functions.take_env_step(net, state, env_easy_2,  epsilon, buffer, device = device) 
  episode_reward += step_reward

  
  if done: # if the episode is done, then log the episode reward

      episode_rewards.append(episode_reward) # holds reward for each game
      mean_reward_100 = np.mean(episode_rewards[-100:])
      
      
      skeleton_helper_functions.print_and_save_logs(writer, steps_count, epsilon, 
                                                    mean_reward_100, 
                                                    episode_reward,
                                                    len(episode_rewards))
      
      if max_mean_reward is None or max_mean_reward < mean_reward_100:
          torch.save(net.state_dict(), 'models/'+agent_type+'/'+ENV_NAME+'_diff_'+'curriculum'+"-best_%.0f.dat" % mean_reward_100)
          
          if max_mean_reward is not None:
              print("Best reward updated %.3f -> %.3f" % (max_mean_reward, mean_reward_100))
          max_mean_reward = mean_reward_100
          
      
      # the episode is finished so reset the env and the episode reward
      episode_reward = 0.0
      state = env_easy_2.reset()
      
      
      
  """
  Load the buffer first!!!!! before synchning or back-propagating and learning on data
  """
  
  if len(buffer) < REPLAY_BUFFER_START_SIZE_FOR_LEARNING:
      print("Populating buffer")
      continue


  if steps_count % NETS_SYNCH_TARGET == 0:
      print('Nets synched')
      target_net.load_state_dict(net.state_dict())



  optimizer.zero_grad()
  
  batch = buffer.sample(BATCH_SIZE)
  
  batch = buffer.to_pytorch_tensors(batch,device)
  
  
  batch_loss = None

  
  if agent_type == "DQN":
        
        batch_loss = skeleton_dqn_architectures.simple_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
        
        
  elif agent_type == "DoubleDQN" or agent_type == "DuelingDoubleDQN" : 
        
        batch_loss = skeleton_dqn_architectures.double_dqn_loss(batch, net, target_net, 
                                                      gamma = GAMMA_FACTOR,
                                                      batch_size = BATCH_SIZE, 
                                                      device = device)
  batch_loss.backward()
  
  optimizer.step()



print('Training finished')

















































