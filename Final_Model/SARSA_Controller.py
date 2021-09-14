"""
SARSA Controller

Train a value function using the SARSA controller with the tile coding approximation

Inputs : name of previous value function, 
        name of new value function, 
        number of episodes, 
        steps per episode,
        action frequency in seconds,
        exploration rate,
        alpha,
        beta

Outputs : Creates new value function weights pickle file in Value Function directory,
        Plots Network durnig training
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
import random

import itertools
from Microgrids_Network_env_fix import *
from vf_tile_coding_approx import ValueFunction


class SarsaController:
    def __init__(self, exp_rate=0.12, beta=0.01): #reward very high, lower value with beta
     
        b = [[0,1], [0,1], [0,1], [0,1], [0,1]]
        self.actions = list(itertools.product(*b))
        self.state = [[0,0,0], [0,0,0], [0,0,0],[0,0,0], [0,0,0], 0,0, 0,0,0,0,0 ]  # see above

        self.exp_rate = exp_rate
        self.beta = beta
        
        #stabilizer test
        self.old_action = random.choice(self.actions)

    def chooseAction(self, new_state, valueFunc):
       
        v_best = float('-inf')
        action = []

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = random.choice(self.actions)
        
        else:
            for a in self.actions:
                v = valueFunc.value(new_state, a)
                if v > v_best:
                    v_best = v
                    action = a

        return action

    def run(self, network, valueFunc, steps=1000, inner_steps=100, idle_iter=50, debug=False ):
        
        avg_reward = 0
        total_reward = 0

        #Random initialization
        network.reset()
        cur_action = random.choice(self.actions)
        cur_state = network.state

        for i in range(1, steps + 1):
            
            mean_reward = 0
            #take average reward during idle_iter 
            for k in range(0, idle_iter): #step the network several times so it inputs a control at most every 1 ms (otehrwise too fast, exp growth)
                new_state, reward, _,_  = network.step(cur_action)
                mean_reward += reward

            mean_reward /= idle_iter

            new_action = self.chooseAction(new_state, valueFunc)

            total_reward += reward
            if debug:
                print("state {} action {} reward {}".format(cur_state, cur_action, reward))
            if i % inner_steps == 0:
                print("step {} -> avg reward {} total reward {}".format(i, avg_reward, total_reward))

            delta = reward - avg_reward + valueFunc.value(new_state, new_action) - valueFunc.value(cur_state,
                                                                                                   cur_action)
            avg_reward += self.beta * delta
            valueFunc.update(cur_state, cur_action, delta)

            cur_state = new_state
            cur_action = new_action

            #Stopping criterion if V too high
            highest_V = 0.0

            for j in range(0,4):
                if(cur_state[j] > highest_V):
                    highest_V = cur_state[j]
       
            if(highest_V >= 120.0):
                print('Explored out of bounds, next episode')
                print('i : ', i, 'highest V : ', highest_V, 'average_reward : ', avg_reward)
                break


def train_sarsa(prev_vf = None , new_vf = 'new_vf_weights', episode_num = 20, episode_steps = 90, 
                action_delay = 2, exploration_rate = 0.12, alpha = 0.01, beta = 0.01):
    
    print('Starting training with SARSA for {}'.format(new_vf))

    #convert time to iterations
    action_freq = int(action_delay/T)

    # Initialize Network
    network = MicrogridsNetwork()
    episode = 0

    sa = SarsaController(exp_rate= exploration_rate , beta = beta)
    vf = ValueFunction(alpha)

    #Use previous value function
    if(prev_vf != None):
        with open('Value_Functions/{}.p'.format(prev_vf), 'rb') as pfile:
            vf.weights = pickle.load(pfile)

    while(episode < episode_num):

        sa.run(network, vf, steps=episode_steps, inner_steps=150, idle_iter = action_freq, debug=False)
        episode +=1
        print('episode ' , episode, 'time : ', "{:.2f}".format(episode*episode_steps*action_freq*T),' seconds')

    #Store value function using pickles
    with open('Value_Functions/{}.p'.format(new_vf), 'wb') as pfile:
        pickle.dump(vf.weights, pfile)
    
    network.plot(timetoplot = int(episode_num*episode_steps*action_freq*T) )

