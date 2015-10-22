from mjcmdp import *
import numpy as np

mdp = WalkerMDP()
viewer = mdp.get_viewer()

ob = mdp.reset()

obslist = []
rewlist = []
for i in xrange(300):
    obs, rewards, dones = mdp.step(np.random.randn(mdp.ctrl_dim) )
    obslist.append(obs)
    rewlist.append(rewards)
    viewer.loop_once()
    if dones: break
print

obsarr = np.concatenate(obslist)
rewarr = np.array(rewlist)
import matplotlib.pyplot as plt
plt.plot(obsarr)
plt.legend(range(obsarr.shape[1]))
plt.plot(rewarr)
plt.show()