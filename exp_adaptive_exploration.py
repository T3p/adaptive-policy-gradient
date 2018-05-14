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
import continuous_acrobot
import mass
import fast_utils

from gym.utils import seeding

from utils import maybe_make_dir, zero_fun

import argparse


from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize


AVAILABLE_EXPERIMENTS = {
        'AdamOnlyTheta': adaptive_exploration.AdamOnlyTheta,
        'MonotonicOnlyTheta' : adaptive_exploration.MonotonicOnlyTheta,
        'MonotonicNaiveGradient' : adaptive_exploration.MonotonicNaiveGradient,
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
        env_name = 'MountainCarContinuous-v0',
        name='',
        batch_size=100,
        max_iters = 10000,
        filepath='experiments',
        random_seed = 0,
        parallel=False,
        verbose=False,
        confidence = 1,
        sigma = 1,
        theta = np.array([0, 0]),
        alpha = None,
        initial_budget = 0
        ):
    print(experiment_class, name, batch_size, max_iters, random_seed)
    #Task
    if alpha is None:
        print("ALPHA IS NONE")
        meta_selector = BudgetMetaSelector(confidence=confidence)
    else:
        print("ALPHA IS NOT NONE")
        meta_selector = ConstMeta(alpha=alpha, N=None, coordinate=False)

    # env = gym.make(env_name)
    # # Tweak for mountain car
    # if 'env' in env.__dict__:
    #     env = env.env
    #R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    gamma = 0.99
    H = 1000#env.horizon

    try:
        tp = TaskProp(gamma,H,env.min_action,env.max_action)
    except:
        try:
            tp = TaskProp(gamma,H,-env.max_action,env.max_action)
        except:
            tp = TaskProp(gamma, H, -10, 10)
    # R = 0
    # M = np.linalg.norm(np.array([env.max_position, env.min_position]), np.inf)

    # tp = TaskProp(
    #         gamma,
    #         H,
    #         -env.max_action,
    #         env.max_action,
    #         R,
    #         M,
    #         env.min_position,
    #         env.max_position,
    #         2*env.max_action
    # )
    local = True

    #Policy
    theta_0 = theta
    np.random.seed(random_seed)
    #theta_0 = np.random.normal(0, 0.1, 4)
    #w = np.array([[math.log(1), 0], [0, math.log(1)]])#math.log(env.sigma_controller)
    w = np.array([math.log(sigma)])
    pol = ExpGaussPolicy(theta_0,w)

    #Features

    if env_name == 'MountainCarContinuous-v0':
        feature_fun = fast_utils.normalize_mountain_car
    elif (env_name == 'ContCartPole-v0') or (env_name == 'ContCartPoleRLLab'):
        feature_fun = fast_utils.normalize_cartpole
    elif env_name == 'RLLAB:Cartpole':
        # feature_fun = fast_utils.normalize_cartpole_rllab
        feature_fun = utils.identity
    else:
        feature_fun = utils.identity



    #Constraints
    constr = OptConstr(
                delta = 0.9,
                N_min=batch_size,
                N_max=500000,
                N_tot = 30000000,
                max_iter = max_iters,
                approximate_gradients=True
    )

    #Evaluation of expected performance
    def evaluate(pol,deterministic=False):
        var = 0 if deterministic else pol.cov
        # return env.computeJ(pol.theta_mat,var)
        return utils.calc_J(pol.theta_mat[0,0], 0.9, 0.9, gamma, var, 4.0, 1)
    # evaluate = zero_fun
    #Run
    # exp = adaptive_exploration.Experiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name)
    # exp = adaptive_exploration.CollectDataExperiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name)
    # exp = adaptive_exploration.SafeExperiment(env, tp, grad_estimator, meta_selector, constr, feature_fun, evaluate, name=name, random_seed=random_seed)

    experiment = AVAILABLE_EXPERIMENTS[experiment_class]
    exp = experiment(env_name, tp, meta_selector, constr, feature_fun, evaluate=evaluate, name=name, random_seed=random_seed, initial_budget=initial_budget)

    exp.run(pol, local, parallel, verbose=verbose, filename=os.path.join(filepath, name + utils.generate_filename()), gamma=1)



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
    parser.add_argument('--random_seed', dest='random_seed', default=seeding.hash_seed() % 2**32, type=int, help='Random seed')
    parser.add_argument('--experiment_class', dest='experiment_class', default=list(AVAILABLE_EXPERIMENTS.keys())[0], type=str, help='type of experiment: ' + ', '.join(AVAILABLE_EXPERIMENTS.keys()))
    parser.add_argument('--env_name', dest='env_name', type=str, default='LQG1D-v0', help='Name of gym environment')
    parser.add_argument('--confidence', dest='confidence', type=int, default=1, help='Multiply every step size by confidence')

    parser.add_argument('--sigma', dest = 'sigma', type=float, default=1, help="Value for sigma")
    parser.add_argument('--theta', dest='theta', type=str, default = '[0,0]', help="Value for theta")

    parser.add_argument('--alpha', dest='alpha', type=str, default=None, help='Fixed step size')
    parser.add_argument('--initial_budget', dest='initial_budget', type=float, default=0., help="Initial budget")
    args = vars(parser.parse_args())

    args['theta'] = np.array(eval(args['theta']))

    maybe_make_dir(args['filepath'])

    run(**args)
