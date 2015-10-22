from atari import AtariMDP
from ale_python_interface import ALEInterface
import random
import cv2
# import matplotlib
# import matplotlib.pyplot as plt

mdp = AtariMDP('atari_roms/pong.bin')
states, _ = mdp.sample_initial_states(1)

init_state = states[0]

action = 0

actions= mdp.action_set

# wait for a while...
for _ in xrange(20000):
    states, obs, rewards, dones = mdp.step(states, [action])
    cv2.imshow("atari",states[0].get_image())
    key = cv2.waitKey(-1)
    idx = key - ord('0')
    print idx
    if idx < len(actions):
        action = idx
    else:
        action = 0
