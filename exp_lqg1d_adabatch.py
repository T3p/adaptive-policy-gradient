import gym
import numpy as np
from lqg1d import LQG1D
import adabatch
from meta_optimization import TaskProp, Estimator, OptConstr, MetaOptimizer
from policies import GaussPolicy
import utils


def run(delta,bound_name,estimator_name,emp=True,parallel=True,filename='record.h5',verbose=True):
    #Task
    env = gym.make('LQG1D-v0')
    R = np.asscalar(env.Q*env.max_pos**2+env.R*env.max_action**2)
    M = env.max_pos
    gamma = env.gamma
    H = env.horizon
    tp = TaskProp(
            R,
            M,
            gamma,
            H,
            min_state = -env.max_pos,
            max_state = env.max_pos,
            min_action = -env.max_action,
            max_action = env.max_action,
            volume = 2*env.max_action
    )
    local = True

    #Policy
    theta_0 = 0
    sigma = env.sigma_controller
    pol = GaussPolicy(theta_0,sigma**2)

    #Features
    phi = utils.identity

    #Gradient estimation
    grad_estimator = Estimator(estimator_name)

    #Constraints
    constr = OptConstr(
                delta,
                N_min=100,
                N_max=500000,
                N_tot = 30000000
    )

    #Meta optimization
    meta_selector = MetaOptimizer(bound_name,constr,estimator_name,emp)

    #Evaluation of expected performance
    def eval_lqg(pol):
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
    run(
        delta = 0.99,
        bound_name = 'bernstein',
        estimator_name = 'gpomdp',
        emp = True,
        parallel = False
    )
        
