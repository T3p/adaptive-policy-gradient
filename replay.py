import gym
import numpy as np
from policies import ExpGaussPolicy
from utils import *
import lqg1d

theta_file = 'theta.npy'
sigma = 0.2
task = 'LQG1D-v0'
episodes = 100000
feat = identity
min_a = -10
max_a = 10

theta = -0.1#np.load(theta_file)
pol = ExpGaussPolicy(-0.1,0)
env = gym.make(task)

avg_ret = 0
for ep in range(episodes):
    done = False
    s = env.reset()
    
    ret = 0
    h = 0
    while not done and h<20:
        #env.render()
        phi = feat(np.ravel(s))
        a = np.clip(pol.act(phi),min_a,max_a)
        s,r,done,_ = env.step(a)
        ret+=0.99**h * r
        h+=1

    avg_ret += 1/(ep+1)*(ret - avg_ret)
    print(avg_ret)