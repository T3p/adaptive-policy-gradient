import gym
import numpy as np
from lqg1d import LQG1D
import adabatch
from meta_optimization import *
from policies import GaussPolicy, ExpGaussPolicy
import utils
import adaptive_exploration
import math


def run(estimator_name='gpomdp',meta_selector=VanishingMeta(alpha=1e-4,N=100),parallel=True,filename='record.h5',verbose=True):
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
    w = math.log(1)#math.log(env.sigma_controller)
    pol = ExpGaussPolicy(theta_0,w)

    #Features
    feature_fun = utils.identity

    #Gradient estimation
    grad_estimator = Estimator(estimator_name)

    #Constraints
    constr = OptConstr(
                delta = 0.1,
                N_min=100,
                N_max=500000,
                N_tot = 30000000,
                max_iter = 10000
    )

    #Evaluation of expected performance
    def evaluate(pol,rewards):
        return env.computeJ(pol.theta_mat,pol.cov)

    #Run
    exp = adaptive_exploration.Experiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate)

    exp.run(pol, local, parallel, verbose=verbose, filename='experiments_non_exact/' + utils.generate_filename())

if __name__ == '__main__':
    #Vanilla
    #run(estimator_name = 'gpomdp', meta_selector = VanishingMeta(1e-3,100), parallel = False)

    #Adabatch
    run(parallel = False)
