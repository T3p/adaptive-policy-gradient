#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:55:14 2018

@author: matteo
"""

import lqg
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAX_ITER = int(1e6)

#Task
env = gym.make('LQG-v0')
R = float(np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2))
M = float(env.max_pos)
gamma = float(env.gamma)
H = env.horizon
volume = float(2*env.max_action)


#Initial setting
theta = -0.1
sigma = 1.
J = env.computeJ(theta, sigma)
budget = 0.
c = (R*M**2)/((1 - gamma)**2*sigma**2)* \
        (volume/np.sqrt(2*np.pi*sigma**2) + \
            gamma/(2*(1 - gamma)))

log = np.zeros((MAX_ITER, 4))

for iteration in range(MAX_ITER):
 
        
    #Update
    grad_J = env.grad_K(theta, sigma)
    #budget -= .25/c #Cost for not being greedy
    #alpha = .5/c #Greedy-safe step
    alpha = .5/c * (1 + np.sqrt(max(0., 1 + 4*c*budget/grad_J**2))) #Largest-safe step
    theta += alpha * grad_J    
    J_new = env.computeJ(theta, sigma)
    budget = J_new - J
    
    #Log
    if iteration%10000 == 0:
        print('Iteration', iteration)
    log[iteration] = [theta, J, budget, alpha]
    
    J = J_new
    
#Saving    
log = pd.DataFrame(log, columns=['theta', 'J', 'budget', 'alpha'])
log.to_csv('log.csv')

#Plotting
log['theta'].plot()
    