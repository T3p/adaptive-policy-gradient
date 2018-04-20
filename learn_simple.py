import gym
import numpy as np
from lqg1d import LQG1D
import adabatch
from meta_optimization import *
from policies import GaussPolicy, ExpGaussPolicy
import utils
import adaptive_exploration
import math
import os
import mass

from gym.utils import seeding

from utils import maybe_make_dir, zero_fun

import argparse



def run(env_name = 'MountainCarContinuous-v0',
        name='',
        batch_size=100,
        max_iters = 10000,
        filepath='experiments',
        random_seed = 0,
        parallel=False,
        verbose=False,
        alpha=0.001
        ):
    print(name, batch_size, max_iters, random_seed)
    #Task
    meta_selector = ConstMeta(alpha=alpha, N=None, coordinate=True)
    env = gym.make(env_name)
    if 'env' in env.__dict__:
        env = env.env

    gamma = 0.999
    H = 1000

    try:
        tp = TaskProp(gamma,H,env.min_action,env.max_action)
    except:
        tp = TaskProp(gamma,H,-env.max_action,env.max_action)

    local = True

    #Policy
    theta_0 = np.array([0, 1])
    w = np.array([math.log(1)])
    pol = ExpGaussPolicy(theta_0,w)

    #Features
    feature_fun = utils.identity

    #Constraints
    constr = OptConstr(
                delta = 0.2,
                N_min=batch_size,
                N_max=500000,
                N_tot = 30000000,
                max_iter = max_iters,
                approximate_gradients=False
    )

    evaluate = zero_fun

    exp = adaptive_exploration.MonotonicOnlyTheta(env_name, tp, meta_selector, constr, feature_fun, evaluate=evaluate, name=name, random_seed=random_seed)

    exp.run(pol, local, parallel, verbose=verbose, filename=os.path.join(filepath, name + utils.generate_filename()))



if __name__ == '__main__':
    #Vanilla
    #run(estimator_name = 'gpomdp', meta_selector = VanishingMeta(1e-3,100), parallel = False)

    #Adabatch
    parser = argparse.ArgumentParser(description='Launch constant learning rate experiments')
    parser.add_argument('--parallel', dest='parallel', default=False, help='Launch parallel',  action='store_true')
    parser.add_argument('--verbose', dest='verbose',  default=False, help='Verbose', action='store_true')
    parser.add_argument('--name', dest='name',default='Exp Budget', help='Identifier for the experiment')
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int, help='Specify batch size')
    parser.add_argument('--max_iters', dest='max_iters', default=2000, type=int, help='Maximum number of iterations')
    parser.add_argument('--filepath', dest='filepath', default='experiments', type=str, help='Where to save the data')
    parser.add_argument('--random_seed', dest='random_seed', default=seeding.create_seed(), type=int, help='Random seed')
    parser.add_argument('--env_name', dest='env_name', type=str, default='LQG1D-v0', help='Name of gym environment')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    maybe_make_dir(args.filepath)

    run(**vars(args))
