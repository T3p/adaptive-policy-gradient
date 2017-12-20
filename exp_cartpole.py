import gym
import numpy as np
import cartpole
import adabatch
from meta_optimization import *
from policies import GaussPolicy
from gradient_estimation import performance
import utils


def run(estimator_name='gpomdp',meta_selector=ConstMeta(alpha=1e-2*np.ones(4),N=100),parallel=True,filename='record.h5',verbose=True):
    #Task
    env = gym.make('ContCartPole-v0')
    tp = TaskProp(
            gamma=0.99,
            H=200,
            min_action = -10,
            max_action = 10,
    )
    local = True

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
                N_min=2,
                N_max=500000,
                N_tot = 30000000,
                max_iter = 200
    )

    #Undiscounted performance
    def eval_cartpole(pol,rewards):
        return performance(rewards)

    #Run
    adabatch.learn(env,tp,pol,phi,constr,
        grad_estimator,
        meta_selector,
        local,
        eval_cartpole,
        parallel,
        'results/' + filename,
        verbose
    )


if __name__ == '__main__':    

    #Vanilla
    run(
        meta_selector = ConstMeta(np.ones(4)*1e-2,N = 100),
        parallel = False
    )

    #Adabatch
    #run(
    #    meta_selector = MetaOptimizer(),
    #    parallel=False)
