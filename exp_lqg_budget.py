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

from utils import maybe_make_dir

import argparse

def run(estimator_name='gpomdp',
        meta_selector=VanishingMeta(alpha=1e-4,N=100),
        parallel=True,
        verbose=True,
        name='',
        batch_size=100,
        max_iters = 10000,
        filepath='experiments'):
    #Task
    meta_selector = BudgetMetaSelector()
    env = gym.make('LQG1D-v0')
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M = env.max_pos
    gamma = env.gamma
    H = env.horizon
    tp = TaskProp(
            gamma,
            H,
            -env.max_action,
            env.max_action,
            R,
            M,
            -env.max_pos,
            env.max_pos,
            2*env.max_action
    )
    local = True

    #Policy
    theta_0 = -0.1
    w = math.log(0.01)#math.log(env.sigma_controller)
    pol = ExpGaussPolicy(theta_0,w)

    #Features
    feature_fun = utils.identity

    #Gradient estimation
    grad_estimator = Estimator(estimator_name)

    #Constraints
    constr = OptConstr(
                delta = 0.1,
                N_min=batch_size,
                N_max=500000,
                N_tot = 30000000,
                max_iter = max_iters
    )

    #Evaluation of expected performance
    def evaluate(pol,rewards):
        return env.computeJ(pol.theta_mat,pol.cov)

    #Run
    exp = adaptive_exploration.SafeExperimentSemiDetPolicy(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name)

    maybe_make_dir(filepath)
    exp.run(pol, local, parallel, verbose=verbose, filename=os.path.join(filepath, utils.generate_filename()))

if __name__ == '__main__':
    #Vanilla
    #run(estimator_name = 'gpomdp', meta_selector = VanishingMeta(1e-3,100), parallel = False)

    #Adabatch
    parser = argparse.ArgumentParser(description='Launch safe budget experiments')
    parser.add_argument('--parallel', dest='parallel', default=False, help='Launch parallel',  action='store_true')
    parser.add_argument('--verbose', dest='verbose',  default=False, help='Verbose', action='store_true')
    parser.add_argument('--name', dest='name',default='Exp Budget', help='Identifier for the experiment')
    parser.add_argument('--batch_size', dest='batch_size', default=100, type=int, help='Specify batch size')
    parser.add_argument('--max_iters', dest='max_iters', default=2000, type=int, help='Maximum number of iterations')
    parser.add_argument('--filepath', dest='filepath', default='experiments', type=str, help='Where to save the data')

    args = parser.parse_args()
    run(parallel = args.parallel, name=args.name, verbose=args.verbose, batch_size=args.batch_size, max_iters=args.max_iters, filepath=args.filepath)
