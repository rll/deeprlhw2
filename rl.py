"""
Some generic classes that are useful for defining and solving MDPs.
"""


import numpy as np

class MDP(object):
    def step(self, states, actions):
        """        
        s,a -> s', o, r, d

        Inputs
        ------
        states
        actions

        Returns
        -------
        (nextstates, observation, rewards, done)
        """
        raise NotImplementedError
    def plot(self, states, actions=None):
        """
        Plot states and actions.
        Should accept actions=None
        """
        raise NotImplementedError

class Policy(object):
    def step(self, o):
        """
        Return dict including

        required: 
            a : actions
        optional:
            pa : specifies probability distribution that 'a' was sampled from
            [whatever else your learning algorithm will need]
        """
        raise NotImplementedError
    

class ValueFunction(object):
    """
    State-value function v(observation)
    """
    def fit(self, paths):
        """
        paths : a list of dictionaries, which have the keys "rewards" "observations" and "returns"
        returns: None
        """
        raise NotImplementedError
    def predict(self, path):
        """
        path : a dictionary, which has the key "observations"
        returns: a numpy float64 vector giving the predicted value at each timestep
        """
        raise NotImplementedError

class NoValueFunction(ValueFunction):
    """
    Value function that is identically zero
    """
    def fit(self, _paths):
        pass
    def predict(self, path):
        return np.zeros(pathlength(path))


class Serializable(object):
    """
    Objects that are pickled and unpickled via their constructor arguments
    """
    def __init__(self, *args):
        self.args = args
    def __getstate__(self):
        return {"args" : self.args}
    def __setstate__(self, d):
        out = type(self)(*d["args"])
        self.__dict__.update(out.__dict__)

def animate_rollout(mdp, policy, horizon=100, delay=0.05):
    """
    Do rollouts and plot at each timestep
    delay : time to sleep at each step
    """
    import time
    obs = mdp.reset()
    mdp.plot()
    for i in xrange(horizon):
        a = policy.step(obs)["action"]
        obs, _rew, done = mdp.step(a)
        mdp.plot()
        if done:
            print "terminated after %s timesteps"%i
            break
        time.sleep(delay)

def rollout(mdp, policy, max_pathlength):
    """
    Simulate the mdp and policy for max_pathlength steps
    """
    ob = mdp.reset()
    terminated = False

    obs = []
    actions = []
    rewards = []
    pdists = []
    for _ in xrange(max_pathlength):
        obs.append(ob)
        pol_out = policy.step(ob)
        action = pol_out["action"]        
        actions.append(action)
        pdists.append(pol_out.get("pdist",[None]))

        ob, rew, done = mdp.step(action)
        rewards.append(rew.sum())
        if done: 
            terminated = True
            break
    return {"observations" : np.concatenate(obs), "pdists" : np.concatenate(pdists), 
        "terminated" : terminated, "rewards" : np.array(rewards), "actions" : np.concatenate(actions)}

def pathlength(path):
    """
    Number of timesteps in the path
    """
    return path["rewards"].shape[0]

