"""
Testing

Test controllers in the microgrids Environment.
Made to work with tile coding approximation for Value Function.
Value Functions should be put in the Value_Functions folder.
Default testing time is 500 seconds.
A free mode without controller is available for experiments
A dummy controller is also provided

Inputs : name of the value function, 
        number of episodes, 
        time of each episode,
        action frequency in seconds

Outputs : Prints reward and time during testing,
        Plots Network behavior and weights of value function at the end

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import random
from TileCoding import *
from Microgrids_Network_env_fix import *
from vf_tile_coding_approx import ValueFunction

# 2 possible actions for each DGU :
# PnP = 0
# MPPT = 1
# Charge = 1
b = [[0,1], [0,1], [0,1], [0,1], [0,1]]
possible_actions = list(itertools.product(*b))


def test(vf_name = 'vf_sarsa_v17', episode_num = 1 , episode_time = 500, action_delay = 2):

    print('Starting test for {}'.format(vf_name))

    # Initialize Network and Value Function
    network = MicrogridsNetwork()
    vf = ValueFunction(network)

    #Use trained value function
    with open('Value_Functions/{}.p'.format(vf_name), 'rb') as pfile:
        vf.weights = pickle.load(pfile)

    #Plot weights of Value Function
    tile_index= np.arange(len(vf.weights))       #Create X array
    x = tile_index.reshape(-1,1)

    plt.figure()
    plt.plot(x, vf.weights, 'ro', label='DGU 3')
    plt.title('Weights of Value Function')
    plt.xlabel('Tile Index')
    plt.ylabel('Weight')
    plt.legend(loc='best', fontsize='medium')
    #plt.show()
    
    # Convert time to iterations
    sim_length = int(episode_time/T)
    action_freq = int(action_delay/T)

    action = random.choice(possible_actions)

    network.reset()

    total_reward = 0

    for episode in range(0, episode_num):
        for i in range (sim_length):

            if((i>0) and (i % action_freq == 0)): #new action every action_freq (50'000 steps= 1 second)
                next_state, reward, _, _ = network.step(action)
                v_best = float('-inf')
                
                # Find best action
                for a in possible_actions:
                    v = vf.value(next_state, a)
                    if v > v_best:
                        v_best = v
                        action = a
            
            else: 
                _, reward, _, _ = network.step(action)
            
            total_reward += reward
            
            if((i>0) and (i % 250000 == 0)):#every 5 seconds
                print('Time : ', int(i*T),' seconds')
                print('Step reward :', reward)
        
        print('Episode ', episode)
    
    network.plot(episode_num*episode_time)
    print('Total reward :',total_reward)

def free(time = 300, action_delay = 4):

    print('Starting free running of Network.')

    #add network initialization
    sim_length = int(time/T)
    action_freq = int(action_delay/T)

    # Initialize Network
    network = MicrogridsNetwork()
    network.reset()
    action = [0,0,0,0,0]

    for i in range(sim_length):        
        
        if((i>0) and (i % action_freq == 0) ):
            #action = random.choice(possible_actions)
            action= [0,0,0,0,0]
            #action = [1,1,1,1,1]
            #print('action  ',i,' :',action)
 
        if((i>0) and (i % 2*action_freq== 0)):
            #action = random.choice(possible_actions)
            #action= [0,0,0,0,0]
            action= [1,1,1,1,1]
            #network.reset()
            #print('action  ',i,' :',action)
        
        _, reward, _, _ = network.step(action)

        if((i>0) and (i % 250000 == 0)): #every 5 seconds
            print('Time : ', int(i*T),' seconds')
            print('Step reward :', reward)

    network.plot(timetoplot = T*sim_length )


def dummy(time = 300): #dummy controller    
    #PnP by default, sets to charge if too low, sets to PnP when too high
    print('Starting Dummy Controller')

    #Convert tiem to iterations
    sim_length = int(time*(1/T))

    # Initialize Network
    network = MicrogridsNetwork()

    Soc_lower_range = 20.0
    SoC_upper_range = 80.0
    
    action = [0,0,0,0,0]
    
    total_reward = 0

    for i in range(sim_length): #update network function 

        if(network.DGU3.SoC < Soc_lower_range):
            action[2] = 1 #set battery to charge
        if(network.DGU5.SoC < Soc_lower_range):
            action[4] = 1 #set battery to charge
        if(network.DGU3.SoC > SoC_upper_range):
            action[2] = 0 #set battery to charge
        if(network.DGU5.SoC > SoC_upper_range):
            action[4] = 0 #set battery to charge
        
        if(i% 5000000 == 0): #every 10 seconds
            print('time : ', i*T, 'seconds')

        _, reward, _,_  = network.step(action)

        total_reward += reward
    
    print('Total reward : ',total_reward)
    network.plot(time)

