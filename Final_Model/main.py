"""
Main function to train and test different controllers

"""
from testing import test, free, dummy
from SARSA_Controller import train_sarsa, SarsaController
from Qfit_Controller import train_qfit, QFittingController
from vf_tile_coding_approx import ValueFunction

# Input run mode to choose what to run
# Change parameters directly in function calls
# Parameters are currently set to default values
# time and action frequency is in seconds, steps indicate the number of iterations

# Possible modes : 'sarsa', 'qfit', 'test' , 'free', 'dummy'
run_mode = 'test'

# Training
if(run_mode == 'sarsa'):
    train_sarsa(prev_vf = None , new_vf = 'vf_weights_sarsa_v2.0', episode_num = 20, episode_steps = 90, 
                action_delay = 2, exploration_rate = 0.12, alpha = 0.01, beta = 0.01)

elif(run_mode == 'qfit'):
    train_qfit(prev_vf = None, new_vf_name = 'vf_weights_qfit_v3.0', num_steps = 15, training_size = 1000, action_delay = 0.2, gamma = 0.1)

# Testing
elif(run_mode == 'test'):
    test(vf_name = 'vf_weights_sarsa_v21' , episode_num = 1 , episode_time = 300, action_delay = 0.2)

elif(run_mode == 'free'):
    free(time = 300, action_delay = 4)

elif(run_mode == 'dummy'): 
    dummy(time = 300)

# Check valid mode
else: 
    print('NOT A RUN MODE')