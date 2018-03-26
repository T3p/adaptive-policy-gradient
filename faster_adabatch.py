#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 13:59:17 2018

@author: matteo
"""

import gym
import numpy as np
import lqg1d #register environment
from policies import GaussPolicy
from interaction import trajectory, clip_action

N_0 = 100
MAX_EPISODES = 10e6
MAX_ITERS = 1#np.infty
THETA_0 = -0.1
SIGMA_0 = 1.
HORIZON = 10


env = gym.make('LQG1D-v0')
pol = GaussPolicy(THETA_0, SIGMA_0)
R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
M = env.max_pos
gamma = env.gamma
a_min = -env.max_action
a_max = env.max_action
volume = 2*env.max_action

n = N_0
iteration = 0
n_tot = 0
while iteration<MAX_ITERS and n_tot<MAX_EPISODES:
    #Collect data
    states = []
    actions = []
    rewards = []
    for k in range(n):
        s, a, r = trajectory(env, 
                             pol,
                             HORIZON,
                             action_fun=lambda a: clip_action(a, a_min, a_max), 
                             )
        states.append(s)
        actions.append(a)
        rewards.append(r)
        
    states = np.stack(states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)

    #Compute performance and gradient statistics
    
    iteration+=1
    n_tot+=n