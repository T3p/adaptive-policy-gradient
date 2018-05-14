#parallelism
import sys
import random
import policies
import gym
from scipy import stats
import lqg1d
import cartpole
import cartpole_rllab
import continuous_acrobot
from joblib import Parallel,delayed
import multiprocessing
import tempfile,os
from gym.utils import seeding           # Use this to generate seeds

#adabatch
from policies import GaussPolicy
from meta_optimization import *
from gradient_estimation import performance, Estimator, Estimators

import time
import signal
import random

from utils import zero_fun, identity, generate_filename, split_batch_sizes
import pandas as pd
import json
import os

import fast_utils
from fast_utils import step_mountain_car

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

SAVE_FREQ = 100

#Trajectory (can be run in parallel)
def trajectory_serial(env, tp, pol, feature_fun, batch_size, initial=None, noises=[], deterministic=False, fast_step = None):
    trace_length = np.zeros((batch_size,))

    features = np.zeros((batch_size, tp.H, pol.feat_dim))
    actions = np.zeros((batch_size, tp.H, pol.act_dim))
    rewards = np.zeros((batch_size, tp.H))

    # s = env.reset([-3.5])
    for n in range(batch_size):
        s = env.reset()
        #s = env.state = [-0.5, 0]
        noises = np.random.normal(0,1,tp.H)
        trace_length[n] = tp.H
        for l in range(tp.H):
            phi = feature_fun(np.ravel(s))
            #a = np.clip(pol.act(phi,noises[l], deterministic=deterministic),tp.min_action,tp.max_action)
            a = np.array([pol.act(phi,noises[l], deterministic=deterministic)])
            if fast_step is not None:
                s,r,done,_ = fast_step(s, a)
            else:
                s,r,done,_ = env.step(a)

            features[n,l] = np.atleast_1d(phi)
            actions[n,l] = np.atleast_1d(a)
            rewards[n,l] = r

            if(done):
                trace_length[n] = l
                break

    # features = stats.zscore(features, axis=1)
    # actions = stats.zscore(actions, axis=1)
    # features = (features - np.min(features, axis=1, keepdims=True)) / (np.max(features, axis=1, keepdims=True) - np.min(features, axis=1, keepdims=True))
    # actions = (actions - np.min(actions, axis=1, keepdims=True)) / (np.max(actions, axis=1, keepdims=True) - np.min(actions, axis=1, keepdims=True))

    if batch_size > 0:
        # scores_theta = apply_along_axis2(pol.score,2,actions,features)
        # scores_sigma = apply_along_axis2(pol.score_sigma,2,actions,features)

        scores_theta = fast_utils.fast_calc_score_theta(actions, features, pol.theta_mat, np.asscalar(pol.inv_cov))
        scores_sigma = fast_utils.fast_calc_score_sigma(actions, features, pol.theta_mat, np.asscalar(pol.cov))

    else:
        scores_theta = np.zeros((batch_size, tp.H, pol.feat_dim)).squeeze()
        scores_sigma = np.zeros((batch_size, tp.H))

    return features, actions, rewards, scores_theta, scores_sigma, trace_length



#Trajectory (can be run in parallel)
def trajectory_parallel(tp, pol, feature_fun, batch_size, initial=None, noises=[], deterministic=False, fast_step = None):
    p = multiprocessing.current_process()
    # return fast_trajectory(p.env, tp, pol, feature_fun, batch_size, initial, noises, deterministic)
    return trajectory_serial(p.env, tp, pol, feature_fun, batch_size, initial, noises, deterministic, fast_step=fast_step)



def process_initializer(q, env_name):
    p = multiprocessing.current_process()
    seed = q.get()
    #p.env = env
    # p.env = gym.make('LQG1D-v0')
    # p.env = gym.make(env_name)
    np.random.seed(seed % 2**32)
    random.seed(seed)
    if env_name.startswith('RLLAB:'):
        p.env = normalize(CartpoleEnv())
        #p.env.wrapped_env.pole.inertia = 0.25   # @CHANGED
    else:
        p.env = gym.make(env_name)
        if 'env' in p.env.__dict__:
            p.env = p.env.env
        p.env.seed(seed)

    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)



class BaseExperiment(object):
    def __init__(self,
                env_name,
                task_prop,
                meta_selector,
                constr=OptConstr(),
                feature_fun=identity,
                evaluate=zero_fun,
                name = 'Budget exp',
                random_seed = 1,
                initial_budget = 0):
        self.env_name = env_name
        if self.env_name == 'LQG1D-v0':
            self.fast_step = fast_utils.step_lqg
            # self.fast_step = None
        elif self.env_name == 'MountainCarContinuous-v0':
            self.fast_step = fast_utils.step_mountain_car
        elif self.env_name == 'ContCartPole-v0':
            self.fast_step = fast_utils.step_cartpole
        elif self.env_name == 'ContCartPoleRLLab-v0':
            self.fast_step = fast_utils.step_cartpole_rllab
        elif self.env_name == 'ContinuousAcrobot-v0':
            self.fast_step = fast_utils.step_acrobot
        else:
            self.fast_step = None

        if self.env_name.startswith('RLLAB:'):
            self.fast_step = None
            self.env = normalize(CartpoleEnv())

            #self.env.wrapped_env.pole.inertia = 0.25   # @CHANGED
        else:
            self.env = gym.make(env_name)
            self.env.seed(random_seed)



        self.task_prop = task_prop
        self.meta_selector = meta_selector
        self.constr = constr
        self.feature_fun = feature_fun
        self.evaluate = evaluate

        self.budget = initial_budget
        self.data = []
        self.count = 0
        self.descs = None

        self.n_cores = multiprocessing.cpu_count() // 2
        self.name = name

        self.random_seed = random_seed
        np.random.seed(random_seed % 2**32)
        random.seed(random_seed)


        self.pool = None

    def get_param_list(self, params):
        """Returns a dictionary containing all the data related to an experiment
        """

        try:
            params['experiment'] = str(params['self'])
            del params['self']
        except:
            pass
        params['gamma'] = self.task_prop.gamma
        params['name'] = self.name
        params.update(self.constr.__dict__)
        try:
            params['confidence'] = params['meta_selector'].confidence
        except:
            pass
        for i,t in enumerate(params['policy'].theta_mat.ravel()):
            params['theta_' + str(i)] = t

        params['sigma'] = params['policy'].sigma
        del params['policy']

        return dict(params)

    def _get_trajectories(self, policy, batch_size, parallel=True, deterministic=False):
        if parallel:
            args = [[self.task_prop, policy, self.feature_fun, b, None, [], deterministic, self.fast_step] for b in split_batch_sizes(batch_size, self.n_cores)]
            data = self.pool.starmap(trajectory_parallel, args)

            features = np.concatenate([data[i][0] for i in range(len(data))])
            actions = np.concatenate([data[i][1] for i in range(len(data))])
            rewards = np.concatenate([data[i][2] for i in range(len(data))])
            scores_theta = np.concatenate([data[i][3] for i in range(len(data))])
            scores_sigma = np.concatenate([data[i][4] for i in range(len(data))])
            trace_length = np.concatenate([data[i][5] for i in range(len(data))])
        else:
            features, actions, rewards, scores_theta, scores_sigma, trace_length = trajectory_serial(self.env, self.task_prop, policy, self.feature_fun, batch_size, None, [], deterministic, fast_step=self.fast_step)


        return features, actions, rewards, scores_theta, scores_sigma, trace_length

    def estimate_policy_performance(self, policy, N, parallel=False, deterministic=False, get_min=False):
        """Estimates the policy performance

        Parameters:
        policy -- the policy to be evaluated
        N -- the batch size

        Returns: the performance of the Policy
        """
        _, _, rewards, _, _, _ = self._get_trajectories(policy, N, parallel=parallel, deterministic=deterministic)
        J_hat = performance(rewards, self.task_prop.gamma, average=False)

        if get_min:
            J_hat = np.min(J_hat)
        else:
            J_hat = np.mean(J_hat)

        return J_hat

    def split_trajectory_count(self, N):
        return N//3, N//3, N - 2*(N//3)

    def get_trajectories_data(self, policy, N, parallel=True):
        """Run N trajectories and returns computed data

        Params:
            policy -- The policy to be evaluated
            N -- The batch size (number of trajectories)
            parallel -- Run in multiple cores

        Returns: a tuple (features, actions, rewards, J, gradients)
        """
        features, actions, rewards, scores_theta, scores_sigma, trace_lengths = self._get_trajectories(policy, N, parallel=parallel)
        self.task_prop.update(features, actions, rewards, self.use_local_stats)
        self.estimator.update(self.task_prop)

        J = performance(rewards, self.task_prop.gamma)
        gradients = self.estimator.estimate(features, actions, rewards, scores_theta, scores_sigma, trace_lengths, policy)

        return features, actions, rewards, J, gradients

    def _enable_parallel(self):
        q = multiprocessing.Queue()
        for i in range(self.n_cores):
            q.put(np.random.randint(0,2**32))

        self.pool = multiprocessing.Pool(self.n_cores, initializer=process_initializer, initargs=(q,self.env_name))

    def save_data(self, filename):
        col_names = self.get_checkpoint_description()
        traj_df = pd.DataFrame(self.data, columns=col_names)
        traj_df.to_pickle(filename + '.gzip')

        with open(filename + '_params.json', 'w') as f:
            json.dump({a:str(b) for a,b in self.initial_configuration.items()}, f)

    def _collect_data(self, params):
        def safe_get(name):
            return params[name] if name in params else 0

        data_row = [
            self.count,
            params['gradients']['grad_w'],
            params['gradients']['gradDeltaW'],
            safe_get('prevJ'),
            safe_get('prevJ_det'),
            safe_get('alpha'),
            safe_get('beta'),
            safe_get('N'),
            params['policy'].sigma,
            self.budget,
            safe_get('J_journey'),
            safe_get('J_det_exact')
        ]
        thetas = np.atleast_1d(params['policy'].theta_mat).ravel()
        for t in thetas:
            data_row.append(t)

        grad_thetas = np.atleast_1d(params['gradients']['grad_theta']).ravel()
        for t in grad_thetas:
            data_row.append(t)

        grad_mixeds = np.atleast_1d(params['gradients']['grad_mixed']).ravel()
        for t in grad_mixeds:
            data_row.append(t)

        if self.descs is None:
            descs = ['T','GRAD_W','GRAD_DELTAW','J','J_DET','ALPHA',
                    'BETA','N','SIGMA','BUDGET','J_JOURNEY','J_DET_EXACT']

            for i,_ in enumerate(thetas):
                descs.append('THETA_' + str(i))

            for i,_ in enumerate(grad_thetas):
                descs.append('GRAD_THETA_' + str(i))

            for i,_ in enumerate(grad_mixeds):
                descs.append('GRAD_MIXED_' + str(i))

            self.descs = descs

        return data_row

    def make_checkpoint(self, params):
        data_row = self._collect_data(params)
        self.data.append(data_row)
        self.count += 1

    def get_checkpoint_description(self):
        return self.descs


    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__







class CollectDataExperiment(BaseExperiment):
    def _collect_data(self, params):
        d = super()._collect_data(params)

        return d + [
            params['gradients']['grad_mixed_reinforce'],
            params['gradients']['grad_delta_reinforce'],
            params['gradients']['grad_mixed_gpomdp1'],
            params['gradients']['grad_delta_gpomdp1'],
            params['gradients']['grad_mixed_gpomdp2'],
            params['gradients']['grad_delta_gpomdp2'],
            params['gradients']['grad_mixed_exact'],
            params['gradients']['grad_delta_exact'],
            params['gradients']['grad_mixed_gpomdp1_baseline'],
            params['gradients']['grad_delta_gpomdp1_baseline']
        ]

    def get_checkpoint_description(self):
        return super().get_checkpoint_description() + \
            [
                'grad_mixed_reinforce',
                'grad_delta_reinforce',
                'grad_mixed_gpomdp1',
                'grad_delta_gpomdp1',
                'grad_mixed_gpomdp2',
                'grad_delta_gpomdp2',
                'grad_mixed_exact',
                'grad_delta_exact',
                'grad_mixed_gpomdp1_baseline',
                'grad_delta_gpomdp1_baseline'
            ]


class AdamOnlyTheta(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.0):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min

        adam_selector = AdamOptimizer()

        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)

        #Learning
        iteration = 0; N_tot = 0; J_hat = prevJ
        J_journey = prevJ
        start_time = time.time()

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', prevJ, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
            J_journey += J_hat * N

            delta_theta = adam_selector.select(gradients)
            policy.update(delta_theta)

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)


class MonotonicOnlyTheta(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.0):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min       # Total number of trajectories to take in this iteration
        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)

        #Learning
        iteration = 0; N_tot = 0; J_hat = prevJ
        J_journey = prevJ
        start_time = time.time()

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', prevJ, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
            J_journey += J_hat * N

            alpha, _ , _ = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N, iteration, budget=None)
            policy.update(alpha * gradients['grad_theta'])

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)

class MonotonicNaiveGradient(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.0):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min // 2       # Total number of trajectories to take in this iteration

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)

        #Learning
        iteration = 0; N_tot = 0; J_hat = prevJ
        J_journey = prevJ
        start_time = time.time()

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', prevJ, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
            J_journey += J_hat * N

            alpha, _ , _ = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N, iteration, budget=None)
            policy.update(alpha * gradients['grad_theta'])
            policy.update_w(np.max(alpha) * gradients['grad_w'])

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)

class MonotonicThetaAndSigma(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min       # Total number of trajectories to take in this iteration

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)

        #Learning
        iteration = 0; N_tot = 0
        J_journey = prevJ
        start_time = time.time()

        N1 = N//2
        N2 = N - N1

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', 0, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)
            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', J_det_exact, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += prevJ * N1

            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, budget=None)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N2, parallel=parallel)
            J_journey += prevJ * N2

            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, N2, iteration, budget=None)
            policy.update_w(beta * gradients['gradDeltaW'])

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)

class MonotonicZeroBudgetEveryStep(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min       # Total number of trajectories to take in this iteration

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        J_baseline = prevJ

        #Learning
        iteration = 0; N_tot = 0
        J_journey = prevJ
        start_time = time.time()

        N1 = N//2
        N2 = N - N1

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', 0, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', J_det_exact, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += prevJ * N1

            budget = 0
            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, budget=budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N2, parallel=parallel)
            J_journey += prevJ * N2

            budget = 0
            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, N2, iteration, budget=budget)
            policy.update_w(beta * gradients['gradDeltaW'])

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)

class NoWorseThanBaselineEveryStep(BaseExperiment):
    def run(self, policy, use_local_stats=False, parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = self.constr.N_min       # Total number of trajectories to take in this iteration

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        J_baseline = prevJ

        #Learning
        iteration = 0; N_tot = 0
        J_journey = prevJ
        start_time = time.time()

        N1 = N//2
        N2 = N - N1

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', prevJ, '\t', 0, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', prevJ, '\t', J_det_exact, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += prevJ * N1

            budget = prevJ - J_baseline
            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, budget=budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N2, parallel=parallel)
            J_journey += prevJ * N2

            budget = prevJ - J_baseline
            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, N2, iteration, budget=budget)
            policy.update_w(beta * gradients['gradDeltaW'])

            J_journey /= N
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached\nEnd experiment')
                break

        # SAVE DATA

        self.save_data(filename)

class ExpBudget_NoDetPolicy(BaseExperiment):
    def run(self,
            policy,
            use_local_stats=False,  # Update task prop only with local stats
            parallel=True,
            filename=generate_filename(),
            verbose=False,
            gamma = 1.):

        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        # N1, N2, N3 = self.split_trajectory_count(N)
        N1 = N // 2
        N3 = N - N1

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        prevJ_det = 0

        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = prevJ
        policy_low_variance = policy

        while iteration < self.constr.max_iter:
            iteration+=1
            try:
                J_det_exact = self.env.computeJ(policy.theta_mat, 0)
            except:
                try:
                    J_det_exact = self.evaluate(policy.theta_mat, 0)
                except:
                    J_det_exact = 0
            prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True)

            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N3*(J_hat - prevJ)            # B += J(theta, sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])


            # PERFORM THIRD STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N3, parallel=parallel)
            J_journey += J_hat * N3

            self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma) - J(theta, sigma)
            prevJ = J_hat

            beta, N3, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N3, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])
            # policy.update_w(beta * gradients['gradDeltaW'])


            # COMPUTE OPTIMAL SIGMA_0 USING THE BOUND
            # d = policy.penaltyCoeffSigma(self.task_prop.R, self.task_prop.M, self.task_prop.gamma, self.task_prop.volume)
            # if self.budget / N2 >= -(gradients['grad_w']**2)/(4*d):
            #     beta_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget/N2))/(gradients['grad_w']**2))) / (2 * d)
            #     beta_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget/N2))/(gradients['grad_w']**2))) / (2 * d)
            #     w_det = min(policy.w + beta_minus*gradients['grad_w'], policy.w + beta_plus*gradients['grad_w'])
            #     policy_low_variance = policies.ExpGaussPolicy(np.copy(policy.theta_mat), w_det)
            # else:
            #     policy_low_variance = policy


            # newJ_det = self.estimate_policy_performance(policy_low_variance, N2, parallel=parallel)
            # J_journey += newJ_det * N2

            # self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            # prevJ_det = newJ_det

            # beta, N3, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N3, iteration, self.budget)


            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break

            # def signal_handler(signal, frame):
            #     self.save_data(filename)
            #     sys.exit(0)
            #
            # #Manual stop
            # signal.signal(signal.SIGINT, signal_handler)

        # SAVE DATA

        self.save_data(filename)


    def _collect_data(self, params):
        d = super()._collect_data(params)

        return d + [
            params['policy_low_variance'].sigma
        ]

    def get_checkpoint_description(self):
        return super().get_checkpoint_description() + \
            [
                'sigma_det'
            ]

class ExpBudget_SemiDetPolicy(BaseExperiment):
    def run(self,
            policy,
            use_local_stats=False,  # Update task prop only with local stats
            parallel=True,
            filename=generate_filename(),
            verbose=False,
            gamma=1.):

        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        N1, N2, N3 = self.split_trajectory_count(N)

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        #prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)
        prevJ_det = prevJ

        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = (2*prevJ + prevJ_det)/3
        policy_low_variance = policy

        while iteration < self.constr.max_iter:
            iteration+=1
            #J_det_exact = self.env.computeJ(policy.theta_mat, 0)
            try:
                J_det_exact = self.env.computeJ(policy.theta_mat, 0)
            except:
                try:
                    J_det_exact = self.evaluate(policy.theta_mat, 0)
                except:
                    J_det_exact = 0
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N3*(J_hat - prevJ)            # B += J(theta, sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])


            # PERFORM THIRD STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N3, parallel=parallel)
            J_journey += J_hat * N3

            self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma) - J(theta, sigma)
            prevJ = J_hat

            beta, N3, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N3, iteration, self.budget)
            # policy.update_w(beta * gradients['gradDeltaW'])


            # COMPUTE OPTIMAL SIGMA_0 USING THE BOUND
            d = policy.penaltyCoeffSigma(self.task_prop.R, self.task_prop.M, self.task_prop.gamma, self.task_prop.volume)
            if self.budget / N2 >= -(gradients['grad_w']**2)/(4*d):
                beta_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget/N2))/(gradients['grad_w']**2))) / (2 * d)
                beta_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget/N2))/(gradients['grad_w']**2))) / (2 * d)
                w_det = min(policy.w + beta_minus*gradients['grad_w'], policy.w + beta_plus*gradients['grad_w'])
                policy_low_variance = policies.ExpGaussPolicy(np.copy(policy.theta_mat), w_det)
            else:
                policy_low_variance = policy


            newJ_det = self.estimate_policy_performance(policy_low_variance, N2, parallel=parallel)
            J_journey += newJ_det * N2

            self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det

            # beta, N3, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N3, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])

            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break

            # def signal_handler(signal, frame):
            #     self.save_data(filename)
            #     sys.exit(0)
            #
            # #Manual stop
            # signal.signal(signal.SIGINT, signal_handler)

        # SAVE DATA

        self.save_data(filename)


    def _collect_data(self, params):
        d = super()._collect_data(params)

        return d + [
            params['policy_low_variance'].sigma
        ]

    def get_checkpoint_description(self):
        return super().get_checkpoint_description() + \
            [
                'sigma_det'
            ]

class ExpBudget_DetPolicy(BaseExperiment):
    def run(self,
            policy,
            use_local_stats=False,  # Update task prop only with local stats
            parallel=True,
            filename=generate_filename(),
            verbose=False,
            gamma=1.):

        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        N1, N2, N3 = self.split_trajectory_count(N)

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)


        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = (2*prevJ + prevJ_det)/3

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N3*(J_hat - prevJ)            # B += J(theta, sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, N1, safe = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            newJ_det = self.estimate_policy_performance(policy, N2, parallel=parallel, deterministic=True)
            J_journey += newJ_det * N2

            self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det

            # PERFORM THIRD STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N3, parallel=parallel)
            J_journey += J_hat * N3

            self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma) - J(theta, sigma)
            prevJ = J_hat

            beta, N3, safe = self.meta_selector.select_beta(policy, gradients, self.task_prop, N3, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])




            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break


        # SAVE DATA

        self.save_data(filename)





class SimultaneousThetaAndSigma_half(BaseExperiment):
    def run(self,policy,use_local_stats=False,parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        # N1, N2, N3 = self.split_trajectory_count(N)
        N2 = N//2
        N1 = N - N2

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)


        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = (prevJ + prevJ_det)/2

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            # features, actions, rewards, J_hat2, gradients2 = self.get_trajectories_data(policy, N1//2, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, _, _ = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1//2, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            newJ_det = self.estimate_policy_performance(policy, N2, parallel=parallel, deterministic=True)
            J_journey += newJ_det * N2

            self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det


            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, N1//2, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])



            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break


        # SAVE DATA

        self.save_data(filename)



class SimultaneousThetaAndSigma_two_thirds_theta(BaseExperiment):
    def run(self,policy,use_local_stats=False,parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        # N1, N2, N3 = self.split_trajectory_count(N)
        N2 = N//2
        N1 = N - N2

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)


        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = (prevJ + prevJ_det)/2

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)
            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            # features, actions, rewards, J_hat2, gradients2 = self.get_trajectories_data(policy, N1//2, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, _, _ = self.meta_selector.select_alpha(policy, gradients, self.task_prop, N1//3, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            newJ_det = self.estimate_policy_performance(policy, N2, parallel=parallel, deterministic=True)
            J_journey += newJ_det * N2

            self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det


            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, 2*N1//3, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])



            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break


        # SAVE DATA

        self.save_data(filename)


class SimultaneousThetaAndSigma_two_thirds_sigma(BaseExperiment):
    def run(self,policy,use_local_stats=False,parallel=True,filename=generate_filename(),verbose=False, gamma=1.):
        self.use_local_stats = False
        self.initial_configuration = self.get_param_list(locals())
        self.estimator = Estimators(self.task_prop, self.constr)

        N = N_old = self.constr.N_min       # Total number of trajectories to take in this iteration
        # N1, N2, N3 = self.split_trajectory_count(N)
        N2 = N//2
        N1 = N - N2

        #Multiprocessing preparation
        if parallel:
            self._enable_parallel()

        def signal_handler(signal, frame):
            if self.pool is not None:
                self.pool.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)


        # COMPUTE BASELINES
        features, actions, rewards, prevJ, gradients = self.get_trajectories_data(policy, N, parallel=parallel)
        prevJ_det = self.estimate_policy_performance(policy, N, parallel=parallel, deterministic=True, get_min=False)


        #Learning
        iteration = 0
        N_tot = 0
        start_time = time.time()
        J_hat = prevJ
        J_journey = (prevJ + prevJ_det)/2

        while iteration < self.constr.max_iter:
            iteration+=1
            J_det_exact = self.evaluate(policy, deterministic=True)
            self.make_checkpoint(locals())          # CHECKPOINT BEFORE SIGMA STEP
            if iteration % SAVE_FREQ == 0:
                self.save_data(filename)

            J_journey = 0

            # PRINT
            # if verbose:
            #     if iteration % 50 == 1:
            #         print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
            #     print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', policy.get_theta(), '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)

            if verbose:
                if iteration % 50 == 1:
                    print('IT\tN\t\tJ\t\t\tJ_DET\t\t\tTHETA\t\tSIGMA\t\t\tBUDGET')
                print(iteration, '\t', N, '\t', J_hat, '\t', prevJ_det, '\t', '['+','.join(map(lambda x : str(x)[:6],policy.get_theta())) + ']', '\t', policy.sigma, '\t', self.budget / N, '\t', time.time() - start_time)


            start_time = time.time()

            # PERFORM FIRST STEP
            features, actions, rewards, J_hat, gradients = self.get_trajectories_data(policy, N1, parallel=parallel)
            # features, actions, rewards, J_hat2, gradients2 = self.get_trajectories_data(policy, N1//2, parallel=parallel)
            J_journey += J_hat * N1

            if iteration > 1:
                self.budget += N1*(J_hat - prevJ)            # B += J(theta', sigma') - J(theta, sigma)
                prevJ = J_hat
                self.budget *= gamma


            alpha, _, _ = self.meta_selector.select_alpha(policy, gradients, self.task_prop, 2*N1//3, iteration, self.budget)
            policy.update(alpha * gradients['grad_theta'])

            # PERFORM SECOND STEP
            newJ_det = self.estimate_policy_performance(policy, N2, parallel=parallel, deterministic=True)
            J_journey += newJ_det * N2

            self.budget += N2*(newJ_det - prevJ_det)    # B += J(theta', 0) - J(theta, 0)

            prevJ_det = newJ_det


            beta, _, _ = self.meta_selector.select_beta(policy, gradients, self.task_prop, N1//3, iteration, self.budget)
            policy.update_w(beta * gradients['gradDeltaW'])



            N_old = N
            J_journey /= N
            #Check if done
            N_tot+=N
            if N_tot >= self.constr.N_tot:
                print('Total N reached')
                print('End experiment')
                break


        # SAVE DATA

        self.save_data(filename)
