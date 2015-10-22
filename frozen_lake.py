from rl import MDP
import numpy as np
from hw_utils import colorize

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class FrozenLake(MDP):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    """

    def __init__(self, desc):
        self.desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = self.desc.shape
        self.maxxy = np.array([nrow-1, ncol-1])
        (startx,), (starty,) = np.nonzero(self.desc=='S')
        self.startstate = np.array([startx,starty])
        self.increments = np.array([[0,-1],[1,0],[0,1],[-1,0]])
    def step(self, a):
        self.lastaction = a
        a = (a + np.random.randint(-1,2)) % 4
        self.state = nextrow, nextcol = np.clip(self.state + self.increments[a[0]], [0,0], self.maxxy)
        statetype = self.desc[nextrow, nextcol]
        done = statetype in 'GH'
        ob = self.state2obs(self.state).reshape(1)
        rew = np.array(float(statetype == 'G'))
        return ob,rew,done
    def reset(self):
        self.lastaction = None
        self.state = self.startstate.copy()
        return self.state2obs(self.state).reshape(1)
    def cost_names(self):
        return ("cost",)
    def state2obs(self, state):
        r,c = state
        return self.ncol*r+c
    def plot(self):
        (i,j) = self.state
        desc = self.desc.tolist()
        desc[i][j] = colorize(desc[i][j], "red", highlight=True)        
        print "action: ", ["LEFT","DOWN","RIGHT","UP"][self.lastaction] if self.lastaction is not None else None
        print "\n".join("".join(row) for row in desc)
    @property
    def n_actions(self):
        return 4
    @property
    def n_states(self):
        return self.nrow * self.ncol
