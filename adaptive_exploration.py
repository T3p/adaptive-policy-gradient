#parallelism
import sys
from joblib import Parallel,delayed
import multiprocessing
import tempfile,os

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import performance, Estimator, Estimators

import time
import signal
import tables
import random

from utils import zero_fun, identity, generate_filename
import pandas as pd
import json


#Trajectory (can be run in parallel)
def trajectory(env,tp,pol,feature_fun,traces,n,initial=None,noises=[], deterministic=False):
    if  len(noises)==0:
        noises = np.random.normal(0,1,tp.H)

    s = env.reset(initial)
    # s = env.reset([-3.5])
    for l in range(tp.H):
        phi = feature_fun(np.ravel(s))
        a = np.clip(pol.act(phi,noises[l], deterministic=deterministic),tp.min_action,tp.max_action)
        s,r,done,_ = env.step(a)
        traces[n,l] = np.concatenate((np.atleast_1d(phi),np.atleast_1d(a),np.atleast_1d(r)))
        if(done):
            break

class Experiment(object):
    def __init__(self,
                env,
                task_prop,
                grad_estimator,
                meta_selector,
                constr=OptConstr(),
                feature_fun=identity,
                evaluate=zero_fun):
        self.env = env
        self.task_prop = task_prop
        self.grad_estimator = grad_estimator
        self.meta_selector = meta_selector
        self.constr = constr
        self.feature_fun = feature_fun
        self.evaluate = evaluate

        self.budget = 0
        self.data = []
        self.count = 0

    def get_param_list(self, params):
        """Returns a dictionary containing all the data related to an experiment
        """

        try:
            params['experiment'] = str(params['self'])
            del params['self']
        except:
            pass
        params['gamma'] = self.task_prop.gamma
        params.update(self.constr.__dict__)
        del params['policy']

        return dict(params)



    def __get_trajectories(self, policy, batch_size, parallel=True, deterministic=False):
        if parallel:
            initials = np.random.uniform(self.task_prop.min_state, self.task_prop.max_state, batch_size)
            noises = np.random.normal(0, 1, (batch_size, self.task_prop.H))
            traces = np.memmap(self.traces_path, dtype=float,shape=(batch_size, self.task_prop.H, policy.feat_dim + policy.act_dim + 1), mode='w+')
            Parallel(n_jobs=self.n_cores)(delayed(trajectory)\
                (self.env, self.task_prop, policy, self.feature_fun, traces, n, initials[n], noises[n]) for n in range(batch_size))
        else:
            traces = np.zeros((batch_size, self.task_prop.H, policy.feat_dim + policy.act_dim + 1))
            for n in range(batch_size):
                trajectory(self.env, self.task_prop, policy, self.feature_fun, traces, n, deterministic=deterministic)

        features = traces[:,:,:policy.feat_dim]
        actions = traces[:,:,policy.feat_dim:policy.feat_dim+policy.act_dim]
        rewards = traces[:,:,-1]

        return features, actions, rewards

    def estimate_policy_performance(self, policy, N, parallel=False, deterministic=False, get_min=False):
        """Estimates the policy performance

        Parameters:
        policy -- the policy to be evaluated
        N -- the batch size

        Returns: the performance of the Policy
        """
        _, _, rewards = self.__get_trajectories(policy, N, parallel=parallel, deterministic=deterministic)
        J_hat = performance(rewards, self.task_prop.gamma, average=False)

        if get_min:
            J_hat = np.min(J_hat)
        else:
            J_hat = np.mean(J_hat)

        return J_hat


    def run(self,
            policy,
            use_local_stats=False,  # Update task prop only with local stats
            parallel=True,
            filename=generate_filename(),
            verbose=False):

        initial_configuration = self.get_param_list(locals())
        print(initial_configuration)
        estimator = Estimators(self.task_prop)

        N = N_old = self.constr.N_min

        #Multiprocessing preparation
        if parallel:
            path = tempfile.mkdtemp()
            self.traces_path = os.path.join(path,'traces.mmap')
            self.n_cores = multiprocessing.cpu_count()

        #Initial print
        if verbose:
            print(self.meta_selector)
            print('Start Experiment')
            print()

        # Compute baseline
        #prevJ = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=False, get_min=True)
        features, actions, rewards = self.__get_trajectories(policy, N, parallel=parallel)
        self.task_prop.update(features, actions, rewards, use_local_stats)
        estimator.update(self.task_prop)

        prevJ = performance(rewards, self.task_prop.gamma)
        gradients = estimator.estimate(features, actions, rewards, policy)


        prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)


        #Learning
        iteration = 0
        N_tot = 0
        while iteration < self.constr.max_iter:
            iteration+=1

            # PERFORM FIRST STEP
            #Collect trajectories of stochastic policy
            features, actions, rewards = self.__get_trajectories(policy, N, parallel=parallel)
            self.task_prop.update(features, actions, rewards, use_local_stats)
            estimator.update(self.task_prop)

            if iteration > 1:
                J_hat = performance(rewards, self.task_prop.gamma)
                self.budget += J_hat - prevJ            # B += J(theta, sigma') - J(theta, sigma)
                prevJ = J_hat


            self.make_checkpoint(locals())          # CHECKPOINT BEFORE THETA STEP
            #Print before
            if verbose:
                print('Epoch: ', iteration,  ' N =', N,  ' theta =', policy.get_theta(), ' sigma =', policy.sigma, ' budget =', self.budget)
                start_time = time.time()

            #Gradient statistics
            gradients = estimator.estimate(features, actions, rewards, policy)

            alpha, N, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N_old, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])




            # PERFORM SECOND STEP
            newJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)
            self.budget += newJ_det - prevJ_det    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det

            self.make_checkpoint(locals())          # CHECKPOINT AFTER DETERMINISTIC EVALUATION


            # PERFORM THIRD STEP
            features, actions, rewards = self.__get_trajectories(policy, N, parallel=parallel)
            self.task_prop.update(features, actions, rewards, use_local_stats)
            estimator.update(self.task_prop)

            J_hat = performance(rewards, self.task_prop.gamma)



            self.budget += J_hat - prevJ            # B += J(theta', sigma) - J(theta, sigma)
            prevJ = J_hat

            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if verbose:
                print('UPDATING SIGMA: ', iteration,  ' N =', N,  ' theta =', policy.get_theta(), ' sigma =', policy.sigma, ' budget =', self.budget)
                start_time = time.time()
            #Gradient statistics
            gradients = estimator.estimate(features, actions, rewards, policy)

            beta, N, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N_old, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])





            N_old = N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break

            #Print after
            if verbose:
                print('alpha =', alpha, 'J_det', prevJ_det,  ' J^ =', J_hat)
                print('time: ', time.time() - start_time)
                print()

            #Manual stop
            signal.signal(signal.SIGINT, signal_handler)

        # SAVE DATA

        traj_df = pd.DataFrame(self.data, columns=['T', 'GRAD_THETA', 'GRAD_W', 'GRAD_MIXED', 'GRAD_DELTAW', 'J', 'J_DET', 'J+J_DET', 'ALPHA', 'BETA', 'N', 'THETA', 'SIGMA', 'BUDGET'])
        traj_df.to_pickle(filename + '.gzip')

        print('initial_configuration: ', initial_configuration.keys())
        with open(filename + '_params.json', 'w') as f:
            json.dump({a:str(b) for a,b in initial_configuration.items()}, f)

    def make_checkpoint(self, params):
        data_row = [
            self.count,
            params['gradients']['grad_theta'],
            params['gradients']['grad_w'],
            params['gradients']['grad_mixed'],
            params['gradients']['gradDeltaW'],
            params['prevJ'],
            params['prevJ_det'],
            params['prevJ'] + params['prevJ_det'],
            params['alpha'] if 'alpha' in params else 0,
            params['beta'] if 'beta' in params else 0,
            params['N'],
            np.asscalar(params['policy'].theta_mat),
            params['policy'].sigma,
            self.budget
        ]
        self.data.append(data_row)
        self.count += 1

    def __str__(self):
        return 'Budget exp'

#Handle Ctrl-C
def signal_handler(signal,frame):
    sys.exit(0)
