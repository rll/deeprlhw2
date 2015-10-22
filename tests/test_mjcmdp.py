from mjcmdp import *
import numpy as np

mdp = HopperMDP()
viewer = mdp.get_viewer()


while not viewer.should_stop():
    ob, rew, done = mdp.step(np.random.randn(mdp.ctrl_dim) )
    viewer.loop_once()
