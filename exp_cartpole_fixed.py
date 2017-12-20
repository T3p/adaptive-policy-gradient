import gym
import numpy as np
import cartpole
import adabatch
from meta_optimization import *
from policies import GaussPolicy
import utils


def run(estimator_name='gpomdp',alpha=1e-4,N=100,parallel=True,filename='record.h5',verbose=True):
    #Task
    env = gym.make('ContCartPole-v0')
    tp = TaskProp(
            gamma=0.99,
            H=200,
            min_action = -10,
            max_action = 10,
    )

    #Policy
    theta_0 = np.zeros(4)
    sigma = 0.1
    pol = GaussPolicy(theta_0,sigma**2)

    #Features
    phi = utils.identity

    #Gradient estimation
    grad_estimator = Estimator(estimator_name)

    #Constraints
    constr = OptConstr(
                N_min=1,
                N_max=500000,
                N_tot = 30000000,
                max_iter = 200
    )

    #Meta optimization
    meta_selector = ConstMeta(alpha,N)

    #Evaluation of expected performance
    def eval_lqg(pol):
        return 0

    #Run
    adabatch.learn(env,tp,pol,phi,constr,
        grad_estimator,
        meta_selector,
        True,
        eval_lqg,
        parallel,
        'results/' + filename,
        verbose
    )


if __name__ == '__main__':    
    run(
        estimator_name = 'gpomdp',
        alpha = np.ones(4)*1e-2,
        N = 100,
        parallel = False
    )
        
