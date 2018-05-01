import gym
import numpy as np
from policies import GaussPolicy
from interaction import identity
import drone
import time

#theta_file = 'theta.npy'
task = 'Drone-v0'
episodes = 10
feat = identity

#theta = np.load(theta_file)
env = gym.make(task)
theta = env.computeOptimalParam()
print(theta)
sigma = 0.1
pol = GaussPolicy(theta,sigma**2)
speedup = 10

for ep in range(episodes):
    done = False
    s = env.reset()
    
    t = 0
    ret = 0
    while not done and t<env.horizon:
        time.sleep(env.tau/speedup)
        env._render()
        phi = feat(np.ravel(s))
        a = pol.act(phi)
        s, r, done, _ = env.step(a)
        t+=1
        ret+=r
    print(ret)