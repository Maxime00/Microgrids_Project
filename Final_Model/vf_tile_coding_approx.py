"""
Value Function Approximation 

creates a value function approximation using the tile coding implementation

CARE : Designed for Network with 5 DGUs, two Batteries and 6 lines (as presented in report)

To use wiht any other network, function must be manually adapted

"""
import numpy as np
from Microgrids_Network_env_fix import *
from TileCoding import *

# Parameters relations
# width of tile = possible range / number of tiles per tiling
# resolution = (width of tile/number of tilings)
# current resolution V - 6V    SoC - 5%

class ValueFunction:

    def __init__(self, alpha=0.01, numOfTilings=4, maxSize=32768): 
        self.maxSize = maxSize
        self.numOfTilings = numOfTilings

        # divide step size equally to each tiling
        self.alpha = alpha #/ numOfTilings  # learning rate for each tile

        self.hashTable = IHT(maxSize)

        # weight for each tile
        self.weights = np.zeros(maxSize)

        #values needs scaling to satisfy the tile software 
        # Scale = number of desired tiles per tiling divided by possible range of V and Soc

        self.VScale =5.0 /120.0 # 
        self.SoCScale = 5.0 /100.0  # 


    # get indices of active tiles for given state and action 
    def getActiveTiles(self, current_sol, action):

        V1, V2, V3, V4, V5, DGU3_SoC, DGU5_SoC = current_sol
      
        # Tile coding reauires a list of DISTINCT integers to compute different actions
        # create action list for tile coding

        tile_action = [0]*len(action)

        i = 0
        for j in range(0,len(action)):              
            if(action[j] == 0):
                tile_action[j]=i
            elif(action[j] == 1):
                tile_action[j]=i+1
            i +=2

        activeTiles = tiles(self.hashTable, self.numOfTilings,
                        [self.VScale *V1,self.VScale *V2,self.VScale *V3,self.VScale *V4,self.VScale *V5, self.SoCScale * DGU3_SoC, self.SoCScale*DGU5_SoC],
                        tile_action)

        return activeTiles

    # estimate the value of given state and action
    def value(self, current_sol, action):
        
        activeTiles = self.getActiveTiles(current_sol, action)
        return np.sum(self.weights[activeTiles])  # /self.numOfTilings

    # learn with given state, action and target
    def update(self, current_sol, action, delta):
        
        activeTiles = self.getActiveTiles(current_sol, action)

        delta *= (self.alpha/len(activeTiles))
        for activeTile in activeTiles:
            self.weights[activeTile] += delta

    # very similar to update, but used in different algorithm, input is total value instead of delta
    def set_weights(self, current_sol, action, value):

        activeTiles = self.getActiveTiles(current_sol, action)

        #divide value by number of tiles so when computing back value it's equal - not sure if this is correct
        new_value = value / len(activeTiles)

        for activeTile in activeTiles:
            self.weights[activeTile] = new_value


