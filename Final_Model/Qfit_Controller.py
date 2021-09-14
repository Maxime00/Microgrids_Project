"""
Qfit Controller

Train a value function using the Qfit controller with the tile coding approximation

Inputs : name of previous value function, 
        name of new value function, 
        number of steps, 
        training size,
        action frequency in seconds,
        gamma,

Outputs : Creates new value function weights pickle file,
        Plots Network during training
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import itertools
from Microgrids_Network_env_fix import *
from vf_tile_coding_approx import ValueFunction
from regression import regression_for_tile_weights

class QFittingController:
    def __init__(self, gamma = 0.1):
        
        # Create action space
        b = [[0,1], [0,1], [0,1], [0,1], [0,1]]
        self.actions = list(itertools.product(*b))

        self.state = [[0,0,0], [0,0,0], [0,0,0],[0,0,0], [0,0,0], 0,0, 0,0,0,0,0 ]  # see above

        self.gamma = gamma

    def findBestAction(self,state, valueFunc):
       
        v_best = float('-inf')
        action = []

        for a in self.actions:
            v = valueFunc.value(state, a)
            if v > v_best:
                v_best = v
                action = a
      
        return action

    def run(self, network, valueFunc, steps=1000, inner_steps=1, training_size= 1000, action_delay = 10000, debug=False):
                
        for i in range(1, steps + 1):
            
            #Select random training set
            #set training_vf to previous vf so as not to lose previous explored states values when updating value Function after regression
            training_vf = valueFunc

            for j in range(0, training_size):

                network.reset()
                action = random.choice(self.actions) 

                "Use averaged reward over all the steps and change weights of first random state !"
                "explore model for at least 0,2 sec to see what this action does, one step is not enough"
                avg_reward = 0

                #10 000 steps for system to stabilize and so currents and Vs update to correct values (0,2 secs)
                for k in range(0, 10000 -1):
                   network.step(action)
                
                #find best action after stabilization, then reward and next state
                new_state, reward, _, _ = network.step(action)
                best_action = self.findBestAction(new_state, valueFunc)
                
                #step network for action_delay seconds so we have proper idea of what this action does
                #Compute average reward over all steps
                for step in range(0, action_delay):
                    _, reward, _ ,_ = network.step(action)
                    avg_reward += reward

                avg_reward /= action_delay
                #added value to reduce avg reward 
                value = 0.01 *avg_reward + self.gamma*valueFunc.value(new_state, best_action)
                
                #update training value function
                training_vf.set_weights(new_state, best_action, value)
            
            if i % inner_steps == 0:
                print('step' , i, 'time : ', "{:.2f}".format(i*training_size*action_delay*T),' seconds')

            #Regression 
            valueFunc = regression_for_tile_weights(valueFunc, training_vf)
        
        return valueFunc


def train_qfit(prev_vf = None, new_vf_name = 'new_vf_weights', num_steps = 15, training_size = 1000, action_delay = 0.2, gamma = 0.1): 
   
    print('Starting training with Qfit for {}'.format(new_vf_name))

    #convert time to iterations
    action_freq = int(action_delay/T)  

    # Initialize Network
    network = MicrogridsNetwork()

    qf = QFittingController(gamma = gamma)
    vf = ValueFunction()
    
    #Load Previous Value Function
    if(prev_vf != None):
        with open('Value_Functions/{}.p'.format(prev_vf), 'rb') as pfile:
            vf.weights = pickle.load(pfile)

    new_vf = qf.run(network, vf, steps=num_steps, training_size = training_size, action_delay = action_freq, debug=False)

    #Store value function weights using pickles
    with open('Value_Functions/{}.p'.format(new_vf_name), 'wb') as pfile:
        pickle.dump(new_vf.weights, pfile)
  
    #Plot weights of Value Function
    #Create X array
    tiles_index= np.arange(new_vf.maxSize)
    x = tiles_index.reshape(-1,1)

    plt.figure()
    plt.plot(x, new_vf.weights, 'mo', label='DGU 3')
    plt.title('Weights of Value Function')
    plt.xlabel('Tile Index')
    plt.ylabel('Weight')
    plt.legend(loc='best', fontsize='medium')
    #plt.show()

    network.plot(timetoplot = int(num_steps*training_size*action_freq*T))