"""
Microgrids Network
Developed to be used as an openAI gym environment

Implements a microgrid network with 5 DGUs, 2 batteries and 6 lines
A representation of this network is available in the report

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

    Networks of DGU, linked together by lines
    Implements electrical physics as well as low level controllers for DGUs
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }


    def __init__(self, number_of_DGUs = 5, number_of_batteries = 2, number_of_lines = 6):
        "Create network here, only fixed values"

        self.create_network() #adds DGUs and lines Network

        #Values for plotting
        self.solutions = []
        self.timelist = [] #in seconds
        self.state= None
        self.iteration = 0
        self.sample_ratio = 250

        self.nbmDGU = number_of_DGUs
        self.nbmBat = number_of_batteries

        #Reward parameters
        self.Vupperbound = 63.0
        self.Vlowerbound = 33.0
        self.Vref = 48
        
        self.SoCupperbound = 80.0
        self.SoClowerbound = 20.0
        self.SoCref = 60.0
        
        self.coeffcost = 1
        self.coeffreward = 1

        self.weightSoc= 40
        self.weightV = 0.0001
        self.weightV_cost = 20

        self.reward_weight = 0.1


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
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]), high=np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]), dtype=np.float32)


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
        
        #update DGUs first
        #Inet = injected current  ->current exiting DGU will be subtracted
        # Connections 
        Inet1 = -self.line1.Il - self.line2.Il - self.line3.Il
        Inet2 = self.line1.Il - self.line4.Il - self.line5.Il
        Inet3 = self.line2.Il + self.line4.Il
        Inet4 = self.line3.Il - self.line6.Il
        Inet5 = self.line5.Il + self.line6.Il

        # Update DGUs
        self.DGU1.update(action[0], Inet1)
        self.DGU2.update(action[1], Inet2)
        self.DGU3.update(action[2], Inet3)
        self.DGU4.update(action[3], Inet4)
        self.DGU5.update(action[4], Inet5)
        
        # Update lines
        self.line1.update()
        self.line2.update()
        self.line3.update()
        self.line4.update()
        self.line5.update()
        self.line6.update()

        #Could change this to be current sol, and add processor
        self.state = (self.DGU1.y[0], self.DGU2.y[0], self.DGU3.y[0], self.DGU4.y[0], self.DGU5.y[0], self.DGU3.SoC, self.DGU5.SoC)

        "Store Values for plots"
        
        current_sol = [self.DGU1.y, self.DGU2.y, self.DGU3.y, self.DGU4.y, self.DGU5.y, 
                        self.DGU3.SoC, self.DGU5.SoC, 
                        self.DGU1.Vs, self.DGU2.Vs, self.DGU3.Vs, self.DGU4.Vs, self.DGU5.Vs,
                        action[0], action[1], action[2], action[3], action[4],
                        self.line1.Il, self.line2.Il, self.line3.Il, self.line4.Il, self.line5.Il, self.line6.Il]

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
            elif(abs(int(current_sol[i][0]) - self.Vref) != 0) :
                reward += (1/abs(int(current_sol[i][0]) - self.Vref))*self.coeffreward*self.weightV      

        for j in range(0, self.nbmBat):
            
            if (current_sol[self.nbmDGU+j] < self.SoClowerbound): 
                reward -= abs(current_sol[self.nbmDGU+j] - self.SoClowerbound)*self.coeffcost*self.weightSoc
            if (current_sol[self.nbmDGU+j] > self.SoCupperbound) :
                reward -= abs(current_sol[self.nbmDGU+j] - self.SoCupperbound)*self.coeffcost*self.weightSoc
            if (abs(int(current_sol[self.nbmDGU+j]) - self.SoCref) ==0) :
                reward += 5 * self.coeffreward*self.weightSoc
            elif (abs(int(current_sol[self.nbmDGU+j]) - self.SoCref) !=0):
                reward += (1/abs(int(current_sol[self.nbmDGU+j]) - self.SoCref))*self.coeffreward*self.weightSoc
        
        reward *= self.reward_weight

        return np.array(self.state), reward, False, {}

        

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        #Wait a few iteratiosn  ?? when resetting so system stays stable

        y1 = [np.random.uniform(0, 100),0.0,0.0]
        y2 = [np.random.uniform(0, 100),0.0,0.0]
        y3 = [np.random.uniform(0, 100),0.0,0.0]
        y4 = [np.random.uniform(0, 100),0.0,0.0]
        y5 = [np.random.uniform(0, 100),0.0,0.0]
        
        random_state = [y1, y2, y3, y4, y5, np.random.uniform(0, 100), np.random.uniform(0,100), 45.0, 45.0, 45.0, 45.0, 45.0]

        standardSoC=80.0
        Il0 = 0.0

        self.DGU1.setSol(y1, random_state[7], standardSoC)
        self.DGU2.setSol(y2, random_state[8], standardSoC)
        self.DGU3.setSol(y3, random_state[9], random_state[5])
        self.DGU4.setSol(y4, random_state[10], standardSoC)
        self.DGU5.setSol(y5, random_state[11], random_state[6])

        self.line1.setSol(Il0)
        self.line2.setSol(Il0)
        self.line3.setSol(Il0)
        self.line4.setSol(Il0)
        self.line5.setSol(Il0)
        self.line6.setSol(Il0)

        self.step([0,0,0,0,0])

        # Need this update so lines are updated with same random value as DGUs
        self.line1.update()
        self.line2.update()
        self.line3.update()
        self.line4.update()
        self.line5.update()
        self.line6.update()

        self.state = (self.DGU1.y[0], self.DGU2.y[0], self.DGU3.y[0], self.DGU4.y[0], self.DGU5.y[0], self.DGU3.SoC, self.DGU5.SoC)

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
        _sol1=[]
        _sol2=[]
        _sol3=[]
        _sol4=[]
        _sol5=[]
        _SoC=[]
        _Vs=[]
        _actions = []
        _I_lines = []
        
        for i in range(len(self.solutions)):
            _sol1.append(self.solutions[i][0])
            _sol2.append(self.solutions[i][1])
            _sol3.append(self.solutions[i][2])
            _sol4.append(self.solutions[i][3])
            _sol5.append(self.solutions[i][4])
            temp = [self.solutions[i][5],self.solutions[i][6]]
            _SoC.append(temp) 
            temp1 = [self.solutions[i][7],self.solutions[i][8],self.solutions[i][9],self.solutions[i][10],self.solutions[i][11] ]
            _Vs.append(temp1)
            temp2 = [self.solutions[i][12],self.solutions[i][13],self.solutions[i][14],self.solutions[i][15],self.solutions[i][16]]
            _actions.append(temp2)
            temp3 = [self.solutions[i][17],self.solutions[i][18],self.solutions[i][19],self.solutions[i][20],self.solutions[i][21],self.solutions[i][22]]
            _I_lines.append(temp3)


        sol1 = np.array(_sol1)
        sol2 = np.array(_sol2)
        sol3 = np.array(_sol3)
        sol4 = np.array(_sol4)
        sol5 = np.array(_sol5)
        SoC = np.array(_SoC)
        Vs = np.array(_Vs)
        actions = np.array(_actions)
        I_lines = np.array(_I_lines)
      
        # add threshold lines in plots
        SoC_thresh_20 = [self.SoClowerbound]*len(SoC[:,0])
        SoC_thresh_80 = [self.SoCupperbound]*len(SoC[:,0])
        V_thresh_33 = [self.Vlowerbound]*len(SoC[:,0])
        V_thresh_63 = [self.Vupperbound]*len(SoC[:,0])

        # Plots

        #SOC of batteries
        plt.figure()
        plt.plot(self.timelist[start:end], SoC[start:end,0], 'm', label='DGU 3')
        plt.plot(self.timelist[start:end], SoC[start:end,1], 'g', label='DGU 5' )
        plt.plot(self.timelist[start:end], SoC_thresh_20[start:end], 'r', label='Thresholds')
        plt.plot(self.timelist[start:end], SoC_thresh_80[start:end], 'r')
        plt.title('State of Charge of Batteries')
        plt.xlabel('Time [seconds]')
        plt.ylabel('State of Charge of batteries')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        #Voltage at DGU outputs (V)
        plt.figure()
        plt.plot(self.timelist[start:end], sol1[start:end,0], 'k', label='V1')
        plt.plot(self.timelist[start:end], sol2[start:end,0], 'c', label='V2')
        plt.plot(self.timelist[start:end], sol3[start:end,0], 'm', label='V3')
        plt.plot(self.timelist[start:end], sol4[start:end,0], 'b', label='V4')
        plt.plot(self.timelist[start:end], sol5[start:end,0], 'g', label='V5')
        plt.plot(self.timelist[start:end], V_thresh_33[start:end], 'r', label='Thresholds')
        plt.plot(self.timelist[start:end], V_thresh_63[start:end], 'r')
        plt.title('Voltage at DGU Outputs')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage at output of DGU - V')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()
        
        #Voltage at DGU source
        plt.figure()
        plt.plot(self.timelist[start:end], Vs[start:end, 0], 'k', label='Vs1')
        plt.plot(self.timelist[start:end], Vs[start:end, 1], 'c', label='Vs2')
        plt.plot(self.timelist[start:end], Vs[start:end, 2], 'm', label='Vs3')
        plt.plot(self.timelist[start:end], Vs[start:end, 3], 'b', label='Vs4')
        plt.plot(self.timelist[start:end], Vs[start:end, 4], 'g', label='Vs5')
        plt.title('Voltage at DGU Source')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Voltage at source of DGU - V')
        plt.legend(loc='best', fontsize='medium')

        #Current at DGU source
        plt.figure()
        plt.plot(self.timelist[start:end], sol1[start:end,1], 'k', label='I1')
        plt.plot(self.timelist[start:end], sol2[start:end,1], 'c', label='I2')
        plt.plot(self.timelist[start:end], sol3[start:end,1], 'm', label='I3')
        plt.plot(self.timelist[start:end], sol4[start:end,1], 'b', label='I4')
        plt.plot(self.timelist[start:end], sol5[start:end,1], 'g', label='I5')
        plt.title('Current at DGU Source')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Current at source of DGU - A')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        #Mode for each DGU
        plt.figure()
        plt.plot(self.timelist[start:end], actions[start:end,0], 'k', label='mode DGU1')
        plt.plot(self.timelist[start:end], actions[start:end,1], 'c', label='mode DGU2')
        plt.plot(self.timelist[start:end], actions[start:end,2], 'm', label='mode DGU3')
        plt.plot(self.timelist[start:end], actions[start:end,3], 'b', label='mode DGU4')
        plt.plot(self.timelist[start:end], actions[start:end,4], 'g', label='mode DGU5')
        plt.title('Mode of DGUs (0-PnP, 1-MPPT/Charge)')
        plt.xlabel('Time [seconds]')
        plt.ylabel('Modes of DGUs')
        plt.legend(loc='best', fontsize='medium')
        #plt.show()

        #Lines currents
        plt.figure()
        plt.plot(self.timelist[start:end], I_lines[start:end,0], 'k', label='I_l 1')
        plt.plot(self.timelist[start:end], I_lines[start:end,1], 'c', label='I_l 2')
        plt.plot(self.timelist[start:end], I_lines[start:end,2], 'm', label='I_l 3')
        plt.plot(self.timelist[start:end], I_lines[start:end,3], 'b', label='I_l 4')
        plt.plot( self.timelist[start:end], I_lines[start:end,4], 'g', label='I_l 5')
        plt.plot( self.timelist[start:end], I_lines[start:end,5], 'g', label='I_l 5')
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

    def create_network(self):
        
        #Values from paper
        Vmmp = 48.5         #Voltage at maximum power point
        Vcharge = 42        #Voltage for battery charging

        y0 = [48.0,0.0,0.0]
        Il0 = 0

        #Feedback condiitions 
        # K[0] < 1
        # K[1] < R
        # 0 < K[2] < (K[0]-1)(K[1]-R)/L 

        #params = [R, L, C, load]
        
        # load values : 1-10 amps

        #TODO - Create_DGU which returns a DGU object with RLC load as input
        
        #coeff for gains [0,1]
        coeff = -1.8#-1.8
        coeff2 = -15#-15
        coeff3 = 0.08#0.08

        #Initialise all objects
        #PV1
        params1 = [2.0e-1 , 2.2e-3, 1.8e-3, 3.0]
        
        K1=[]
        K1.append(coeff)
        K1.append(params1[0] *coeff2)
        K1.append(((K1[0]-1)*(K1[1]-params1[0])/params1[1])*coeff3)


        self.DGU1 = DGU("PV",1,  params1, Vmmp, y0, K1)

        #PV2
        params2 = [3.0e-1 , 1.9e-3, 2.0e-3, 5.2]

        K2=[]
        K2.append(coeff)
        K2.append(params2[0]*coeff2)
        K2.append(((K2[0]-1)*(K2[1]-params2[0])/params2[1])*coeff3)

        self.DGU2 = DGU("PV",2 , params2, Vmmp, y0, K2)

        #Battery 
        params3 = [5.0e-1 , 2.5e-3, 3.0e-3, 8.0]
        K3=[]
        K3.append(coeff)
        K3.append(params3[0]*coeff2)
        K3.append(((K3[0]-1)*(K3[1]-params3[0])/params3[1])*coeff3)

        self.DGU3 = DGU("Battery", 3,  params3, Vcharge, y0, K3)

        #PV3
        params4 = [1.0e-1 , 1.7e-3, 2.2e-3, 8.3] 

        K4=[]
        K4.append(coeff)
        K4.append(params4[0] *coeff2)
        K4.append(((K4[0]-1)*(K4[1]-params4[0])/params4[1])*coeff3)

        self.DGU4 = DGU("PV",4,  params4, Vmmp, y0, K4)

        #Battery2
        params5 = [1.5e-1 , 1.8e-3, 2.3e-3, 9.2] 

        K5=[]
        K5.append(coeff)
        K5.append(params5[0] *coeff2)
        K5.append(((K5[0]-1)*(K5[1]-params5[0])/params5[1])*coeff3)

        self.DGU5 = DGU("Battery", 5,  params5, Vcharge, y0, K5)

        #line1
        self.line1 = line(7.0e-2, 2.1e-6, Il0, self.DGU1, self.DGU2)

        #line2
        self.line2 = line(4.0e-2, 2.3e-6, Il0, self.DGU1, self.DGU3)

        #line3
        self.line3 = line(8.0e-2, 1.8e-6, Il0, self.DGU1, self.DGU4)

        #line4
        self.line4 = line(7.0e-2, 2.0e-6, Il0, self.DGU2, self.DGU3)

        #line5
        self.line5 = line(5.0e-2, 2.2e-6, Il0, self.DGU2, self.DGU5)
        
        #line6
        self.line6 = line(6.0e-2, 1.9e-6, Il0, self. DGU4, self.DGU5)
