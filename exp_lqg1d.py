import gym
import numpy as np
from lqg1d import LQG1D
import adabatch
from meta_optimization import *
from policies import GaussPolicy
import utils


def run(estimator_name='gpomdp',meta_selector=VanishingMeta(alpha=1e-4,N=100),parallel=True,filename='record.h5',verbose=True):
    #Task
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
    theta_0 = -0.01
    sigma = env.sigma_controller
    pol = GaussPolicy(theta_0,sigma**2)

    #Features
    phi = utils.identity

    #Gradient estimation
    grad_estimator = Estimator(estimator_name)

    #Constraints
    constr = OptConstr(
                delta = 0.1,
                N_min=100,
                N_max=500000,
                N_tot = 30000000,
                max_iter = 200
    )

    #Evaluation of expected performance
    def eval_lqg(pol,rewards):
        return env.computeJ(pol.theta_mat,pol.cov)

    #Run
    adabatch.learn(env,tp,pol,phi,constr,
        grad_estimator,
        meta_selector,
        local,
        eval_lqg,
        parallel,
        'results/' + filename,
        verbose
    )


if __name__ == '__main__':    
    #Vanilla
    #run(estimator_name = 'gpomdp', meta_selector = VanishingMeta(1e-3,100), parallel = False)
        
    #Adabatch
    run(meta_selector = MetaOptimizer(bound_name='chebyshev'), parallel = False)
