import gym
import numpy as np
from policies import GaussPolicy
from interaction import identity
import mass
import sys
import time

#theta_file = 'theta.npy'
task = 'Mass-v0'
horizon = 100
episodes = 10
feat = identity
min_a = -1.
max_a = 1.

#theta = np.load(theta_file)
env = gym.make(task)
theta = env.computeOptimalParam()
sigma = 0.01
pol = GaussPolicy(theta,sigma**2)

for ep in range(episodes):
    done = False
    s = env.reset()
    
    t = 0
    while not done and t<horizon:
        time.sleep(.1)
        env._render()
        phi = feat(np.ravel(s))
        a = np.clip(pol.act(phi),min_a,max_a)
        s, r, done, _ = env.step(a)
        t+=1