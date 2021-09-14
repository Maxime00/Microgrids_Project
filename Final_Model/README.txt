All code here is implemented to teach an RL agent to control a microgrid

Anyone using this should refer to the report titled 'Data-Driven Supervisory Control of Microgrids' for details

main : can train and test sarsa and qfit controller using the fix network env (and only this type of network)
NAF_Controller : can train and test NAf controller using network env (with any network), also works with fix network and is much faster


All other functions are designed to be called by each other

SARSA_Controller : implements sarsa algorithm and training function
Qfit_Controller : implements qfit algorithm and training function
testing : implements test of RL algorithms, free running of network and dummy controller
Microgrid_Network_env : implements any microgrid network
Microgrid_Network_env : implements fix microgrid network (described in report)
Objects_Definitions : DGU and lines implementation
vf_tile_coding_approx : implements value function tile coding approximation
TileCoding : Tile coding from R. Sutton
regression : implments value function regression for Qfit algortihm

Concerning Value Function storage :
Only the weights array of the Vf is stored
previous trials can be found in the folder Value_Functions
New vf will be automatically stored in that folder after training!

Concerning python environments : 

NAF_Controller is from outdated keras-rl library, requires module versions as follows : 
python = 3.6
keras = 2.2.4
tensorflow = 1.13.1

Other functions run in python 3.7 with latest versions of all following modules in January 2020 :
numpy, matplotlib, pickle, random, itertools, scipy and gym


Maxime Gautier