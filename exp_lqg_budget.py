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

from gym.utils import seeding

from utils import maybe_make_dir, zero_fun

import argparse


AVAILABLE_EXPERIMENTS = {
        'MonotonicOnlyTheta' : adaptive_exploration.MonotonicOnlyTheta,
        'MonotonicThetaAndSigma' : adaptive_exploration.MonotonicThetaAndSigma,
        'MonotonicZeroBudgetEveryStep' : adaptive_exploration.MonotonicZeroBudgetEveryStep,
        'NoWorseThanBaselineEveryStep' : adaptive_exploration.NoWorseThanBaselineEveryStep,
        'ExpBudget_NoDetPolicy' : adaptive_exploration.ExpBudget_NoDetPolicy,
        'ExpBudget_SemiDetPolicy' : adaptive_exploration.ExpBudget_SemiDetPolicy,
        'ExpBudget_DetPolicy' : adaptive_exploration.ExpBudget_DetPolicy,
        'SimultaneousThetaAndSigma_half' : adaptive_exploration.SimultaneousThetaAndSigma_half,
        'SimultaneousThetaAndSigma_two_thirds_theta' : adaptive_exploration.SimultaneousThetaAndSigma_two_thirds_theta,
        'SimultaneousThetaAndSigma_two_thirds_sigma' : adaptive_exploration.SimultaneousThetaAndSigma_two_thirds_sigma
    }



def run(experiment_class='Experiment',
        name='',
        batch_size=100,
        max_iters = 10000,
        filepath='experiments',
        random_seed = 0,
        parallel=False,
        verbose=False,
        ):
    random_seed = 0
    print(experiment_class, name, batch_size, max_iters, random_seed)
    #Task
    meta_selector = BudgetMetaSelector()
    env = gym.make('MountainCarContinuous-v0')
    env = env.env
    #R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    R = 0
    M = np.linalg.norm(np.array([env.max_position, env.min_position]), np.inf)
    gamma = 0.999
    H = 1000#env.horizon
    tp = TaskProp(
            gamma,
            H,
            -env.max_action,
            env.max_action,
            R,
            M,
            env.min_position,
            env.max_position,
            2*env.max_action
    )
    local = True

    #Policy
    theta_0 = np.array([0, 20])
    # theta_0 = np.array([-0.1])
    #w = np.array([[math.log(1), 0], [0, math.log(1)]])#math.log(env.sigma_controller)
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

    #Evaluation of expected performance
    # def evaluate(pol,deterministic=False):
    #     var = 0 if deterministic else pol.cov
    #     return env.computeJ(pol.theta_mat,var)
    evaluate = zero_fun
    #Run
    # exp = adaptive_exploration.Experiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name)
    # exp = adaptive_exploration.CollectDataExperiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name)
    # exp = adaptive_exploration.SafeExperiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name, random_seed=random_seed)

    experiment = AVAILABLE_EXPERIMENTS[experiment_class]
    exp = experiment(env, tp, meta_selector, constr, feature_fun, evaluate=evaluate, name=name, random_seed=random_seed)

    exp.run(pol, local, parallel, verbose=verbose, filename=os.path.join(filepath, name + utils.generate_filename()))



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
    parser.add_argument('--random_seed', dest='random_seed', default=seeding.create_seed(), type=int, help='Random seed')
    parser.add_argument('--experiment_class', dest='experiment_class', default=list(AVAILABLE_EXPERIMENTS.keys())[0], type=str, help='type of experiment: ' + ', '.join(AVAILABLE_EXPERIMENTS.keys()))

    args = parser.parse_args()

    maybe_make_dir(args.filepath)

    run(**vars(args))
