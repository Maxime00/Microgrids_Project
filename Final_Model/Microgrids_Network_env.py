"""
Microgrids Network

Developed to be used as an openAI gym environment

Can create any type of network when initializing, slower than previous implementation

"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

from Objects_definitions import DGU
from Objects_definitions import line
from Objects_definitions import T


class MicrogridsNetwork(gym.Env):

    """
    Description of Env

    Networks of DGUs, linked together by lines
    Implements electrical physics as well as low level control for DGUs
    DGUs have two modes of operation PI (0) and free(1) which are imposed through the action list
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self, types_of_DGUs, connections): 
        """
        Create network when initializing object

        When initializing, specify the type of each DGU in a list 'types_of_DGUs' 
        and the connections between each DGU in a 2D list 'connections'

        Exemple : 
        types_of_DGUs = ['PV', 'PV', 'Battery', 'PV', 'Battery']
        connections = [[1,2], [1,3], [1,4], [2,3], [2,5], [4,5]]
        network1 = MicrogridsNetwork(types_of_DGUs, connections)
        """

        self.DGUs = []
        self.lines = []

        # Number of Elements in Network
        self.nbmDGU = len(types_of_DGUs)
        self.nbmBat = 0

        self.create_network(types_of_DGUs, connections)

        #Values for plotting
        self.solutions = []
        self.timelist = [] #in seconds
        self.state= []
        self.iteration = 0
        self.sample_ratio = 2500

        #Reward parameters
        self.Vupperbound = 63.0
        self.Vlowerbound = 33.0
        self.Vref = 48.0
        
        self.SoCupperbound = 80.0
        self.SoClowerbound = 20.0
        self.SoCref = 60.0
        
        self.coeffcost = 1
        self.coeffreward = 1

        self.weightSoc= 40
        self.weightV = 0.01
        self.weightV_cost = 20

        self.reward_weight = 0.001 #added weight to reduce size of reward at each step

        # Define Action Space (Discrete)
        # 3 solutions for action space rep, tuples, multi binary or multi discrete (could create errors in RL code)
        
        #self.action_space = spaces.MultiDiscrete([2,2,2,2,2]) # 0 =PnP 1=MPPT/Charge
        self.action_space = spaces.MultiBinary(5) # 0 =PnP 1=MPPT/Charge
        #self.action_space = spaces.Tuple(spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)) # 0 =PnP 1=MPPT/Charge

        #Define State Space - (Continuous with 5 voltages and 2 SoC)
        # Min voltage = 0
        # Max Voltage = 100
        # Min SoC = 0
        # Max Soc = 100
        low = np.zeros((self.nbmDGU + self.nbmBat))
        high = np.full((self.nbmDGU + self.nbmBat), 100.0)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Seed for random number generator
        self.seed()

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        #Check action validity
        for i in range (len(action)):
            if((action[i] != 0) and (action[i] != 1)):
                print('Invalid action : ', action )
        
        "Step Network" 

        #update DGUs first
        #Inet = injected current  -> current exiting DGU will be subtracted
        # Connections
        # Calculate Inets
        Inets = []

        for i in range(0, len(self.DGUs)):
            Inet = 0
            for j in range (0, len(self.lines)):
                if(self.lines[j].DGU_in.index == self.DGUs[i].index): # DGU_in gives current to line, subtract from Inet 
                    Inet -= self.lines[j].Il 
                if(self.lines[j].DGU_out.index == self.DGUs[i].index): #DGU_out gets current from line, add to Inet
                    Inet += self.lines[j].Il 
            Inets.append(Inet)

        # Update DGUs
        for i in range(0, len(self.DGUs)):
            self.DGUs[i].update(action[i], Inets[i])
        
        # Update lines
        for i in range(0, len(self.lines)):
            self.lines[i].update()

        # Update State 
        for i in range(0, self.nbmDGU):
            self.state.append(self.DGUs[i].y[0])
        for i in range(0, self.nbmDGU):
            if(self.DGUs[i].category == 'Battery'):
                self.state.append(self.DGUs[i].SoC)

        "Store Values for plots"
        current_sol = []
        
        for i in range(0, self.nbmDGU):  #Add Output Voltages
            current_sol.append(self.DGUs[i].y)
        for i in range(0, self.nbmDGU):  #Add Source Voltages
            current_sol.append(self.DGUs[i].Vs)
        for i in range(0, self.nbmDGU):  #Add actions
            current_sol.append(action[i])
        for i in range(0, self.nbmDGU):  #Add SoC
            if(self.DGUs[i].category == 'Battery'):
                current_sol.append(self.DGUs[i].SoC)
        for i in range(0, len(self.lines)):  #Add line currents
            current_sol.append(self.lines[i].Il)

        if(self.iteration % self.sample_ratio == 0 ):
            self.solutions.append(current_sol)
            self.timelist.append((self.iteration*T))  #add /60 for it to be in minutes

        #step iterations for proper value sampling
        self.iteration +=1

        "Calculate Reward"
        reward = 0

        for i in range(0,self.nbmDGU):
            if (current_sol[i][0] < self.Vlowerbound) :
                reward -= abs(current_sol[i][0] - self.Vlowerbound)*self.coeffcost*self.weightV_cost
            if (current_sol[i][0] > self.Vupperbound) :
                reward -= abs(current_sol[i][0] - self.Vupperbound)*self.coeffcost*self.weightV_cost
            if (abs(int(current_sol[i][0]) - self.Vref) == 0) :
                reward += self.coeffreward*self.weightV
            elif (abs(int(current_sol[i][0]) - self.Vref) != 0) :
                reward += (1/abs(int(current_sol[i][0]) - self.Vref))*self.coeffreward*self.weightV      

        for j in range(3*self.nbmDGU, self.nbmBat):
            
            if (current_sol[j] < self.SoClowerbound): 
                reward -= abs(current_sol[j] - self.SoClowerbound)*self.coeffcost*self.weightSoc
            if (current_sol[j] > self.SoCupperbound) :
                reward -= abs(current_sol[j] - self.SoCupperbound)*self.coeffcost*self.weightSoc
            if (abs(int(current_sol[j]) - self.SoCref) ==0) :
                reward += 10 * self.coeffreward*self.weightSoc
            elif (abs(int(current_sol[j]) - self.SoCref) !=0):
                reward += (1/abs(int(current_sol[j]) - self.SoCref))*self.coeffreward*self.weightSoc
        
        reward *= self.reward_weight

        return np.array(self.state), reward, False, {}

        

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        standardSoC=80.0
        Il0 = 0.0

        #Set DGUs
        for i in range(0, self.nbmDGU):
            if(self.DGUs[i].category == 'PV'):
                rand_y = [np.random.uniform(20, 70),0.0,0.0]
                rand_V = np.random.uniform(43, 53)
                self.DGUs[i].setSol(rand_y, rand_V, standardSoC )
            elif(self.DGUs[i].category == 'Battery'):
                rand_y = [np.random.uniform(20, 70),0.0,0.0]
                rand_V = np.random.uniform(43, 53)
                rand_SoC = np.random.uniform(0, 100)
                self.DGUs[i].setSol(rand_y, rand_V, rand_SoC )

        #Set lines
        for i in range (0, len(self.lines)):
            self.lines[i].setSol(Il0)

        self.step([0,0,0,0,0])

        # Need this update so lines are updated with same random value as DGUs
        for i in range(0, len(self.lines)):
            self.lines[i].update()

        #Update State
        for i in range(0, self.nbmDGU):
            self.state.append(self.DGUs[i].y[0])
        for i in range(0, self.nbmDGU):
            if(self.DGUs[i].category == 'Battery'):
                self.state.append(self.DGUs[i].SoC)

        return np.array(self.state)

    def plot(self, timetoplot):
        """Plot the results from the testing for the desired amount of time

        """
        # find number of iterations ot plot
        iterationstoplot = int(timetoplot/T)
        temp = timetoplot/T
        if((temp-iterationstoplot) != 0.0):
            iterationstoplot +=1

        # Start and end of desired iterations to plot
        start = int((self.iteration - (iterationstoplot))/self.sample_ratio)
        end = int(self.iteration/self.sample_ratio)

        # Convert sols array to plotable arrays
        _V_pcc = []
        _Is = []
        _Vs=[]
        _actions = []
        _SoC=[]
        _I_lines = []

        # Extract solutions from solutions array
        # Restructure to plottable 2D arrays
        for k in range(0,self.nbmDGU) :  # V, Is, Vs and actions
            V_temp = []
            I_temp = []
            Vs_temp = []
            actions_temp = []
            for i in range(len(self.solutions)):
                V_temp.append(self.solutions[i][k][0])
                I_temp.append(self.solutions[i][k][1])
                Vs_temp.append(self.solutions[i][self.nbmDGU+k])
                actions_temp.append(self.solutions[i][2*self.nbmDGU +k])
            _V_pcc.append(V_temp)
            _Is.append(I_temp)
            _Vs.append(Vs_temp)
            _actions.append(actions_temp)

        for k in range(0, self.nbmBat) : #SoC
            SoC_temp = []
            for i in range(len(self.solutions)):
                SoC_temp.append(self.solutions[i][3*self.nbmDGU + k])
            _SoC.append(I_temp)

        for k in range(0, len(self.lines)): # Line currents
            Il_temp = []
            for i in range(len(self.solutions)):
                Il_temp.append(self.solutions[i][3*self.nbmDGU + self.nbmBat + k ])
            _I_lines.append(Il_temp)

        V_pcc = np.array(_V_pcc)
        Is = np.array(_Is)
        Vs = np.array(_Vs)
        actions = np.array(_actions)
        SoC = np.array(_SoC)
        I_lines = np.array(_I_lines)
      
        # add threshold lines in plots
        SoC_thresh_20 = [self.SoClowerbound]*len(SoC[0,:])
        SoC_thresh_80 = [self.SoCupperbound]*len(SoC[0,:])
        V_thresh_33 = [self.Vlowerbound]*len(SoC[0, :])
        V_thresh_63 = [self.Vupperbound]*len(SoC[0, :])

        # Plots
        # SOC of batteries
        plt.figure()
        for i in range(0, self.nbmBat):
            plt.plot(self.timelist[start:end], SoC[i, start:end], label='DGU {}'.format(i))
        plt.plot(self.timelist[start:end], SoC_thresh_20[start:end], 'r', label='Thresholds')
        plt.plot(self.timelist[start:end], SoC_thresh_80[start:end], 'r')
        plt.title('State of Charge of Batteries')
        plt.xlabel('Time [seconds]')
        plt.ylabel('State of Charge [%]')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        # Voltage at DGU outputs (V)
        plt.figure()
        for i in range(0, self.nbmDGU):
            plt.plot(self.timelist[start:end], V_pcc[i, start:end],  label='DGU {}'.format(i))
        plt.plot(self.timelist[start:end], V_thresh_33[start:end], 'r', label='Thresholds')
        plt.plot(self.timelist[start:end], V_thresh_63[start:end], 'r')
        plt.title('Voltage at DGU Outputs (V_pcc)')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage at output of DGU - V')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        # Voltage at DGU source
        plt.figure()
        for i in range(0, self.nbmDGU):
            plt.plot(self.timelist[start:end], Vs[i, start:end], label='DGU {}'.format(i))
        plt.title('Voltage at DGU Source (V_s)')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage at source of DGU - V')
        plt.legend(loc='best', fontsize='medium')

        # Current at DGU Source
        plt.figure()
        for i in range(0, self.nbmDGU):
            plt.plot(self.timelist[start:end], Is[i, start:end], label='DGU {}'.format(i))
        plt.title('Current at DGU Source')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Current at source of DGU - A')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        # Mode for each DGU
        plt.figure()
        for i in range(0, self.nbmDGU):    
            plt.plot(self.timelist[start:end], actions[i, start:end], label='mode DGU {}'.format(i))
        plt.title('Mode of DGUs (0-PnP, 1-MPPT/Charge)')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Modes of DGUs')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        #Lines currents
        plt.figure()
        for i in range(0, len(self.lines)):
            plt.plot(self.timelist[start:end], I_lines[i, start:end], label='line {}'.format(i))
        plt.title('Line Currents')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Current [A]')
        plt.legend(loc='best', fontsize='medium')
        plt.show()


    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()



    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()


    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_network(self, types_of_DGUs, connections):
        """Creates Network
        Called in init to create the network

        """
        #Values for DGUs
        Vmpp = 48.5         #Voltage at maximum power point
        Vcharge = 42        #Voltage for battery charging

        #Initial values 
        y0 = [48.0,0.0,0.0]
        Il0 = 0

        #Feedback conditions 
        # K[0] < 1
        # K[1] < R
        # 0 < K[2] < (K[0]-1)(K[1]-R)/L 

        #params = [R, L, C, load]

        #coeff for gains [0,1]
        coeff = -1.8#-1.8
        coeff2 = -15#-15
        coeff3 = 0.08#0.08

        for i in range(0, len(types_of_DGUs)):

            params = [np.random.uniform(0.1, 1.0) , #Resistance
                        np.random.uniform(1.5e-3, 2.5e-3), #Inductance
                        np.random.uniform(2e-3, 3e-3), #Capacity
                        np.random.uniform(5.0 , 10.0)] #Load
        
            Gains=[]
            Gains.append(coeff)  #K1
            Gains.append(params[0] *coeff2) #K2
            Gains.append(((Gains[0]-1)*(Gains[1]-params[0])/params[1])*coeff3) #K3

            index = i+1

            if(types_of_DGUs[i] == 'PV'):

                DGU_PV = DGU('PV', index, params, Vmpp, y0, Gains)
                self.DGUs.append(DGU_PV)

            elif(types_of_DGUs[i] == 'Battery'):
                
                DGU_Bat = DGU('Battery', index, params, Vcharge, y0, Gains)
                self.DGUs.append(DGU_Bat)
                self.nbmBat += 1

            elif((types_of_DGUs[i] != 'PV') and (types_of_DGUs[i] != 'Battery')):
                print('DGU #', i,' has incorrect type')
        
        for j in range(0, len(connections)):
            # find DGU corresponding to index indicated in connections
            for k in range(self.nbmDGU):
                if (connections[j][0] == self.DGUs[k].index):
                    DGU_in = self.DGUs[k]
                if (connections[j][1] == self.DGUs[k].index):
                    DGU_out = self.DGUs[k]

            new_line = line(np.random.uniform(4e-2, 8e-2), np.random.uniform(1.5e-6, 2.5e-6), Il0, DGU_in, DGU_out)
            self.lines.append(new_line)

        print('Network created with {} DGUs and {} lines'.format(self.nbmDGU, len(self.lines)))