"""
NAF Controller

Implement keras-rl algorithm

"""
import numpy as np
import gym
from Microgrids_Network_env_fix import MicrogridsNetwork

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

#NEED to add plots to see results


# Processor for reward, not sure if useful
class MicrogridsProcessor(Processor):
    def __init__(self):
    
        self.actions = []

    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.

    def process_action(self,action):
        #process action since is algo is not made to handle binary action array
        #convert from float to binary
        new_action = [0.0]* len(action)

        self.actions.append(action)

        for i in range(len(action)):
            if(action[i] >= 0.0):
                new_action[i] = 1
            elif( action[i] <= 0.0):
                new_action[i] = 0

        return new_action


ENV_NAME = 'Microgrid'


# Initialize Network and Value Function
env = MicrogridsNetwork()

# Get the environment and extract the number of actions.
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]


# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(16))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

#shouldn,' need mu model to approximaet policy ?

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(16))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))  #problem is that action output is not in action mode format (0 or 1)
#added own function to activations to have binary output
#added action processor, should work?
#previous - linear
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = MicrogridsProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions) #used to add noise, not desired here since actions are discreet

agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=None,
                 gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

run_mode = 'test'
version = 3

if(run_mode =='train'):
    agent.fit(env, nb_steps=36000, action_repetition=5000, visualize=False, verbose=1, nb_max_episode_steps=None)

    # After training is done, we save the final weights.
    agent.save_weights('Value_Functions/cdqn_{}_weights_v{}.h5f'.format(ENV_NAME,version), overwrite=True)

    print('actions :', processor.actions)
    #ccenv.plot(timetoplot = 1000) 

elif(run_mode =='test'):
    # Load weights
    agent.load_weights('Value_Functions/cdqn_{}_weights_v{}.h5f'.format(ENV_NAME,version))

    # Finally, evaluate our algorithm for 5 episodes.
    agent.test(env, nb_episodes=3, action_repetition=5000 ,visualize=False, nb_max_episode_steps=3000) #

    #Plot testing to see results
    env.plot(timetoplot = 3*300) 

