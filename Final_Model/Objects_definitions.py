"""
Objects Definition

Compatible with openAI gym env, to be used by Microgrids_Network_env

"""

import numpy as np

T = 2.0e-5 #timestep  2.0e-5

#Values for Vs(SoC) dependency and smooth transfer
transfer_steps = 1000       #number of steps for smooth transfer
SoC0_threshold = 2          #threshold for switching to SoC0 mode
SoC100_threshold = 95       #threshold for switching to SoC100 mode
Vtop = 50                   #Voltage when SoC is 100%
Vmin = 3                   #Voltage when SoC is 0%

class DGU:

    def __init__(self, category, index, params, Vmax, y0, K):
        self.category = category  # PV or Battery
        self.index = index        # Index
        self.params = params      # params : Resistance, Inductor, Capacitance and load values
        self.Vmax= Vmax           # Maximum Power Point Voltage or Vcharge depending on category
        self.Vref = 48.0          # Bus Reference Voltage
        self.y = y0               # values in current step
        self.K = K                # Feedback Gain
        self.V = 0.0              # Voltage at output of DGU
        self.SoC = 60.0           # Initial State of Charge
        self.Cbat = 3.5e0         # Capacitor in battery [Ah] 
        self.Vs = self.Vref       # Vs value for plots 
        self.free_Vs = Vmax       # saved Vs value for bumpless transfer
        self.free_SP = Vmax       # Setpoint for bumpless transfer
        self.old_mode = None                    # memory of previous mode for bumpless transfer (0= Pnp, 1=MPPT, 2 = Charge)
        self.SoC_mode = 'SoCNorm'               # battery mode for SoC dependency
        self.SoC_old_mode = 'SoCNorm'           # battery mode for smooth transfer in SoC dependency modes
        self.SoC0_Vs = Vmax                     # saved Vs value for bumpless transfer  
        self.SoC100_Vs = Vmax                   # saved Vs value for bumpless transfer
        self.Vs_at_SoC0_switch = self.Vref      # saved Vs value for SoC dependency
        self.Vs_at_SoC100_switch = self.Vref    # saved Vs value for SoC dependency
        
      
    def update(self, control_mode, Inet):

        # PnP = 0
        # MPPT = 1
        # Charge = 1

        if self.category == "PV":
            if control_mode == 1: #MPPT
                
                #y=[V,Is]
                
                y_next = self.free(Inet)
                self.Vs = self.free_SP

                self.V = self.y[0]  #Voltage to use for line current calculations in same step
                self.y = y_next
                self.old_mode = 1
               
                return y_next

            elif control_mode == 0: #PnP
            
                #y=[V,Is,vi]
        
                y_next = self.PI(Inet)
                
                if(self.old_mode != 0):
                    self.Vs = self.free_Vs
                else :
                    self.Vs = self.K[0]*self.y[0] + self.K[1]*self.y[1] + self.K[2]*self.y[2]
                
                self.V = self.y[0]
                self.y = y_next
                self.old_mode = 0
                
                return y_next


            else:
                print("This is not a control mode")
        
        elif self.category == "Battery":
            
            if control_mode == 1 : #Charge

                #y=[V,Is]
               
                self.getSoC(control_mode)               
                   
                y_next = self.free(Inet)
                self.Vs = self.free_Vs #self.Vs only used for plots

                #Voltage to use for line current calculations in same step
                self.V = self.y[0]  
                self.y = y_next
                
                #For smooth mode transfer
                self.old_mode = 1

                return y_next

            elif control_mode == 0: #PnP
                
                #y=[V,Is,vi]

                self.getSoC(control_mode)
                 
                y_next = self.PI(Inet)

                #Voltage to use for line current calculations in same step
                self.V = self.y[0]
                
                self.y = y_next
                
                #For smooth mode transfer
                self.old_mode = 0
            
                return y_next

            else:
                print('This is not a control mode', control_mode)


        else:
            print("This is not a DGU")


    def setMode(self, control_mode):
        self.control_mode = control_mode
    
    def setVmax(self, Vmax):
        self.Vmax=Vmax
        
    def setGain(self, K):
        self.K= K
    
    def setCategory(self, category):
        self.category = category
    
    def setSol(self, y, V, SoC):
        self.y = y
        self.V = V
        self.SoC = SoC

    def free (self, Inet):
    
        V=self.y[0]
        Is=self.y[1]
        vi =  0

        R = self.params[0]
        L = self.params[1]
        C = self.params[2]
        load = self.params[3]

        if(self.SoC_mode == 'SoC0'):
            
            Vs = self.SoC *((self.Vs_at_SoC0_switch - Vmin)/SoC0_threshold) + Vmin

            V_next = V + T*((Is-load+Inet)/C)
            
            Is_next = Is + T*((-V - R*Is +Vs)/L)

            self.SoC_old_mode = 'SoC0'

            #bumpless transfer - set setpoint in PI to current V
            self.free_Vs = Vs
        
        elif(self.SoC_mode == 'SoC100'):

            Vs = self.SoC * ((Vtop - self.Vs_at_SoC100_switch)/(100 - SoC100_threshold)) + ((100*self.Vs_at_SoC100_switch - Vtop*SoC100_threshold)/(100 - SoC100_threshold))         

            V_next = V + T*((Is-load+Inet)/C)
            
            Is_next = Is + T*((-V - R*Is +Vs)/L)

            self.SoC_old_mode = 'SoC100'

            #bumpless transfer - set setpoint in PI to current V
            self.free_Vs =  Vs

        elif(self.SoC_mode == 'SoCNorm'):

            V_next = V + T*((Is-load+Inet)/C)
            
            #model with i_old 
            Is_next = Is + T*((-V - R*Is +self.free_SP)/L)
            
            #bumpless transfer - free setpoint grows gradually closer to desired setpoint
            err = self.Vmax - self.free_SP
            self.free_SP += err/transfer_steps

            self.SoC_old_mode = 'SoCNorm'

            #bumpless transfer - set setpoint in PI to current V
            self.free_Vs =  self.free_SP 
        
        self.Vs = self.free_Vs

        y_next = [V_next, Is_next, vi]

        return y_next
   
    def PI(self, Inet):
        
        V=self.y[0]
        Is=self.y[1]
        vi=self.y[2]

        R = self.params[0]
        L = self.params[1]
        C = self.params[2]
        load = self.params[3]

        if(self.SoC_mode == 'SoC0'):
            
            Vs = self.SoC*((self.Vs_at_SoC0_switch-Vmin)/ SoC0_threshold) + Vmin

            V_next = V + T*((Is-load+Inet)/C)
            
            Is_next = Is + T*((-V - R*Is +Vs)/L)

            vi_next = (Vs - self.K[0]*V - self.K[1]*Is)/self.K[2]
            
            #For smooth mode transfer
            self.SoC_old_mode = 'SoC0'
            self.SoC0_Vs = Vs

            self.Vs = Vs

        elif(self.SoC_mode == 'SoC100'):

            Vs = self.SoC * ((Vtop - self.Vs_at_SoC100_switch)/(100 - SoC100_threshold)) + (100*self.Vs_at_SoC100_switch - Vtop*SoC100_threshold)/(100-SoC100_threshold)

            V_next = V + T*((Is-load+Inet)/C)
            
            Is_next = Is + T*((-V - R*Is +Vs)/L)

            vi_next = (Vs - self.K[0]*V - self.K[1]*Is)/self.K[2]

            #For smooth mode transfer
            self.SoC_old_mode = 'SoC100'
            self.SoC100_Vs = Vs
            
            self.Vs = Vs

        elif(self.SoC_mode == 'SoCNorm'):

            V_next = V + T*((Is-load+Inet)/C)
            
            #model with i_old 
            Is_next = Is + T*(((self.K[0]-1)*V/L)+((self.K[1]-R)*Is/L)+(self.K[2]*vi/L))
            
            #bumpless transfer from all other modes
            if((self.old_mode != 0) and (self.SoC_old_mode == 'SoCNorm')):
                vi_next = (self.free_Vs - self.K[0]*V - self.K[1]*Is)/self.K[2]
            
            elif((self.old_mode == 0) and (self.SoC_old_mode == 'SoCNorm')):
                vi_next = vi + T*(self.Vref-V)
            
            if(self.SoC_old_mode == 'SoC0'):    
                vi_next = (self.SoC0_Vs - self.K[0]*V - self.K[1]*Is)/self.K[2]
            
            if(self.SoC_old_mode == 'SoC100'):    
                vi_next = (self.SoC100_Vs - self.K[0]*V - self.K[1]*Is)/self.K[2]
            
            #For smooth mode transfer
            self.SoC_old_mode = 'SoCNorm'

            self.Vs = self.K[0]*V_next + self.K[1]*Is_next + self.K[2]*vi_next

        #bumpless transfer - set setpoint in free to currrent Vs
        self.free_SP  = self.K[0]*V_next + self.K[1]*Is_next + self.K[2]*vi_next
        
        y_next =[V_next, Is_next, vi_next]
        
        return y_next

    def getSoC(self, control_mode):

        #Approximate Coulomb counting method
        self.SoC -= (T*self.y[1])/self.Cbat

        #SoC Limits
        if(self.SoC <= 0.0):
            self.SoC = 0.0
                    
        if(self.SoC > 100.0):
            self.SoC = 100.0

        #Soc Dependency
        if(self.SoC <= SoC0_threshold):
            self.SoC_mode = 'SoC0'
            self.Vs_at_SoC0_switch = self.Vs
        
        if(self.SoC >= SoC100_threshold):
            self.SoC_mode = 'SoC100'
            self.Vs_at_SoC100_switch = self.Vs

        if((self.SoC_mode == 'SoC0') and (self.SoC >= (SoC0_threshold))):
            self.SoC_mode = 'SoCNorm'

        if((self.SoC_mode == 'SoC100') and (self.SoC <= (SoC100_threshold))):
            self.SoC_mode = 'SoCNorm'


class line:

    def __init__(self, R, L, Il0, DGU_in, DGU_out):
        self.R=R
        self.L = L
        self.Il = Il0       #current value 
        self.DGU_in = DGU_in
        self.DGU_out = DGU_out
        
    def update(self):
        
        #model with i_old
        Il_next = self.Il + T*(-(self.R/self.L)*self.Il + self.DGU_in.V/self.L - self.DGU_out.V/self.L)

        self.Il = Il_next

    def setSol(self, I):
        self.Il = I    