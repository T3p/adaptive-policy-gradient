#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:04:37 2018

@author: matteo
"""
import numpy as np

#Feature functions
def identity(s):
    return np.ravel(s)

#Action modification functions
def clip_action(a, a_min, a_max):
    return np.clip(a, a_min, a_max)

#Trajectory (fixed horizon)
def trajectory(env, pol, horizon, feature_fun=identity, action_fun=None):
    feats = []
    actions = []
    rewards = []
    
    s = env.reset()
    done = False
    t = 0
    while not done and t<horizon: 
        phi = feature_fun(s)
        a = pol.act(phi)
        a_mod = a if action_fun is None else action_fun(a)
        s, r, done, _ = env.step(a_mod)
        feats.append(phi)
        actions.append(np.atleast_1d(a))
        rewards.append(r)
        t+=1
    
    return np.stack(feats), np.stack(actions), np.stack(rewards)