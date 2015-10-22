from rl import MDP, Serializable
import numpy as np
import sys
sys.path.append("vendor/Arcade-Learning-Environment")
from ale_python_interface import ALEInterface #pylint: disable=F0401

OBS_RAM = 0
OBS_IMAGE = 1



def to_rgb(ale):
    (screen_width,screen_height) = ale.getScreenDims()
    arr = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
    ale.getScreenRGB(arr)
    # The returned values are in 32-bit chunks. How to unpack them into
    # 8-bit values depend on the endianness of the system
    if sys.byteorder == 'little': # the layout is BGRA
        arr = arr[:,:,0:3].copy() # (0, 1, 2) <- (2, 1, 0)
    else: # the layout is ARGB (I actually did not test this.
          # Need to verify on a big-endian machine)
        arr = arr[:,:,2:-1:-1]
    img = arr
    return img

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class AtariMDP(MDP, Serializable):

    def __init__(self, rom_path, obs_type=OBS_RAM, frame_skip=4):
        Serializable.__init__(self, rom_path, obs_type, frame_skip)
        self.options = (rom_path, obs_type, frame_skip)
        
        self.ale = ALEInterface()
        self.ale.loadROM(rom_path)        
        self._rom_path = rom_path
        self._obs_type = obs_type
        self._action_set = self.ale.getMinimalActionSet()
        self.frame_skip = frame_skip


    def get_image(self):
        return to_rgb(self.ale)
    def get_ram(self):
        return to_ram(self.ale)
    def game_over(self):
        return self.ale.game_over()
    def reset_game(self):
        return self.ale.reset_game()

    @property
    def n_actions(self):
        return len(self.action_set)

    def get_obs(self):
        if self._obs_type == OBS_RAM:
            return self.get_ram()[None,:]
        else:
            assert self._obs_type == OBS_IMAGE
            return self.get_image()[None,:,:,:]

    def step(self, a):

        reward = 0.0
        action = self.action_set[a]
        for _ in xrange(self.frame_skip):
            reward += self.ale.act(action)
        ob = self.get_obs().reshape(1,-1)
        return ob, np.array([reward]), self.ale.game_over()

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self.get_obs()

    @property
    def action_set(self):
        return self._action_set

    def plot(self):
        import cv2
        cv2.imshow("atarigame",self.get_image()) #pylint: disable=E1101
        cv2.waitKey(10) #pylint: disable=E1101
