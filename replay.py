import gym
import numpy as np
from policies import GaussPolicy
from utils import *
import cartpole

theta_file = 'theta.npy'
sigma = 1
task = 'ContCartPole-v0'
episodes = 1
feat = identity
min_a = -10
max_a = 10

theta = np.load(theta_file)
pol = GaussPolicy(theta,sigma**2)
env = gym.make(task)

for ep in range(episodes):
    done = False
    s = env.reset()
    
    while not done:
        env.render()
        phi = feat(np.ravel(s))
        a = np.clip(pol.act(phi),min_a,max_a)
        s,r,done,_ = env.step(a)

