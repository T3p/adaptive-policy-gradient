from enum import Enum
import lqg1d
import numpy as np
import math
import itertools
import os
import utils
import json
import random
import string
from utils import range_unlimited, maybe_make_dir
import pandas as pd

try:
    import numba
    NUMBA_PRESENT = True
except ImportError:
    NUMBA_PRESENT = False


MIN_SIGMA = 0.01
MAX_SIGMA = 30

SIGMA_ITERATIONS = 1000000
SIGMA_EPS = 1e-03


LQG_ENV = lqg1d.LQG1D()

#
# Compute constants
#

OPTIMAL_K = np.asscalar(LQG_ENV.computeOptimalK())


class SigmaFunction(object):
    """This is an abstract function with one parameter, used to update parameter sigma
    """
    def __init__(self, param, initial_param):
        self.param = param
        self.initial_param = param if initial_param is None else param
        self.description = 'sigma'


    def update(self, step_size, gradient_sigma):
        """Performs the update x <- x + step_size * gradient_sigma
        """
        raise NotImplemented('Function not implemented')

    def eval(self):
        raise NotImplemented('Function not implemented')

    def __str__(self):
        return self.description


class Identity(SigmaFunction):
    """Implements function f(x) = x
    """
    def __init__(self, param, initial_param = None):
        super().__init__(param, initial_param)
        self.description = str(param if initial_param is None else param)

    def update(self, step_size, gradient_sigma):
        new_param = self.param + step_size*gradient_sigma
        new_param = max(new_param, MIN_SIGMA)

        return Identity(new_param)

    def eval(self):
        return self.param

class Fixed(SigmaFunction):
    """Implements a constant function
    """
    def __init__(self, param, initial_param = None):
        super().__init__(param, initial_param)
        self.description = 'F' + str(param if initial_param is None else param)

    def update(self, step_size, gradient_sigma):
        return self

    def eval(self):
        return self.param

class Exponential(SigmaFunction):
    """Implements function f(x) = e^x
    """
    def __init__(self, param, initial_param=None):
        super().__init__(param, initial_param)
        self.description = 'e(' + str(param if initial_param is None else param) + ')'

    def update(self, step_size, gradient_sigma):
        new_param = self.param + step_size * gradient_sigma #* math.exp(self.param)
        new_param = min(new_param, math.log(MAX_SIGMA))
        new_param = max(new_param, math.log(MIN_SIGMA))

        #self.param = new_param
        return Exponential(new_param, self.initial_param)

    def eval(self):
        return math.exp(self.param)

    def __str__(self):
        return self.description








class Experiment(object):
    """Defines an experiment, with possibly different update rules
    """
    def __init__(self, lqg_environment, exp_name = None):
        self.lqg_environment = lqg_environment
        self.exp_name = exp_name

        #   Store data for logging
        self.t = 0
        self.UPDATE_DIM = 10000
        self.num_rows = self.UPDATE_DIM
        self.num_updates = 0

        self.J_data = np.zeros(self.UPDATE_DIM)
        self.theta_data = np.zeros(self.UPDATE_DIM)
        self.sigma_data = np.zeros(self.UPDATE_DIM)
        self.grad_K_data = np.zeros(self.UPDATE_DIM)
        self.gradSigma_data = np.zeros(self.UPDATE_DIM)
        self.gradMixed_data = np.zeros(self.UPDATE_DIM)
        self.exploration_data = np.zeros(self.UPDATE_DIM)
        #
        #   COMPUTE CONSTANTS RELATED TO LQG_ENVIRONMENT
        #

        self.M = lqg_environment.max_pos
        self.ENV_GAMMA = lqg_environment.gamma
        self.ENV_VOLUME = 2*lqg_environment.max_action
        self.ENV_R = np.asscalar(lqg_environment.R)
        self.ENV_Q = np.asscalar(lqg_environment.Q)
        self.ENV_B = np.asscalar(lqg_environment.B)
        self.ENV_MAX_ACTION = lqg_environment.max_action

        self.MAX_REWARD = self.ENV_Q * self.M**2 + self.ENV_R * self.ENV_MAX_ACTION**2

        self.C1 = (1 - self.ENV_GAMMA)**3 * math.sqrt(2 * math.pi)
        self.C2 = self.ENV_GAMMA * math.sqrt(2 * math.pi) * self.MAX_REWARD * self.M**2
        self.C3 = 2*(1 - self.ENV_GAMMA) * self.ENV_VOLUME * self.MAX_REWARD * self.M**2

        self.m = 1


    def on_sigma_update(self, current_theta, current_sigma_fun):
        """Perform an update on sigma
        """
        return current_sigma_fun


    def on_theta_update(self, theta, sigma_fun):
        """Perform an update on theta
        """
        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        alpha=1/(2*c)

        return theta + alpha*self.gradK


    def on_before_update(self, current_theta, current_sigma_fun):
        """Compute gradients, log data
        """
        # Compute the gradients
        current_sigma = current_sigma_fun.eval()

        self.J = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, current_sigma, self.M, self.ENV_B)
        self.gradK = utils.calc_K(current_theta, current_sigma, self.ENV_GAMMA, self.ENV_R, self.ENV_Q, self.M)
        self.gradSigma = utils.calc_sigma(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA)

        self.gradW = self.gradSigma * math.exp(current_sigma_fun.param)

        self.gradMixed = utils.calc_mixed(self.ENV_GAMMA, current_theta, self.ENV_R, self.ENV_Q)

        self.gradMixedW = self.gradMixed * math.exp(current_sigma_fun.param)

        # Save trajectory data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.num_rows:
                self.J_data = np.concatenate([self.J_data, np.zeros(self.UPDATE_DIM)])
                self.theta_data = np.concatenate([self.theta_data, np.zeros(self.UPDATE_DIM)])
                self.sigma_data = np.concatenate([self.sigma_data, np.zeros(self.UPDATE_DIM)])
                self.grad_K_data = np.concatenate([self.grad_K_data, np.zeros(self.UPDATE_DIM)])
                self.gradSigma_data = np.concatenate([self.gradSigma_data, np.zeros(self.UPDATE_DIM)])
                self.gradMixed_data = np.concatenate([self.gradMixed_data, np.zeros(self.UPDATE_DIM)])
                self.exploration_data = np.concatenate([self.exploration_data, np.zeros(self.UPDATE_DIM)])

                self.num_rows += self.UPDATE_DIM

            self.J_data[self.num_updates] = self.J
            self.theta_data[self.num_updates] = current_theta
            self.sigma_data[self.num_updates] = current_sigma
            self.grad_K_data[self.num_updates] = self.gradK
            self.gradSigma_data[self.num_updates] = self.gradSigma
            self.gradMixed_data[self.num_updates] = self.gradMixed
            performance_deterministic = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.exploration_data[self.num_updates] = performance_deterministic - self.J

    def on_after_update(self, current_theta, current_sigma_fun):
        if self.t % self.downsample == 0:
            self.num_updates += 1

        self.t += 1


    def get_traj_data(self):
        """Returns a tuple containing data matrix and a vector of column description
        """
        data = np.hstack([  np.arange(0, self.t, self.downsample).reshape(-1,1),
                            self.J_data[:self.num_updates].reshape(-1,1),
                            self.theta_data[:self.num_updates].reshape(-1,1),
                            self.sigma_data[:self.num_updates].reshape(-1,1),
                            self.grad_K_data[:self.num_updates].reshape(-1,1),
                            self.gradSigma_data[:self.num_updates].reshape(-1,1),
                            self.gradMixed_data[:self.num_updates].reshape(-1,1),
                            self.exploration_data[:self.num_updates].reshape(-1, 1)])
        columns = ['T', 'J', 'THETA', 'SIGMA', 'GRAD_K', 'GRAD_SIGMA', 'GRAD_MIXED', 'EXPLORATION']
        return data, columns


    def get_param_list(self, params):
        """Returns a dictionary containing all the data related to an experiment
        """
        try:
            params['experiment'] = str(params['self'])
            del params['self']
        except:
            pass
        return params


    def run(self,
            theta=-0.1,
            sigma_fun=Exponential(0),
            n_iterations=-1,
            eps=1e-03,
            filename=None,
            verbose=True,
            two_steps=True,
            downsample=1):
        """Runs the experiment.
        """
        initial_configuration = self.get_param_list(locals())
        self.downsample = downsample

        for t in range_unlimited(n_iterations):
            sigma = sigma_fun.eval()
            # Early stopping if converged or diverged
            if abs(OPTIMAL_K - theta) <= eps:# and abs(sigma) <= MIN_SIGMA:
                break
            if abs(theta) > 10:
                print("DIVERGED")
                break

            sigma = sigma_fun.eval()
            self.on_before_update(theta, sigma_fun)

            # Display
            if verbose:
                if t % 100 == 0:
                    print("\nT\t\tTheta\t\tSigma\t\tJ\t\tJ(0)\n")
                print("{}\t\t{:.5f}\t\t{:.10f}\t\tJ{:.5f}\t\tJ(0){:.5f}\t\t{:.5f}".format(t, theta, sigma, self.J, utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B), self.budget))

            # UPDATE PARAMETERS

            if two_steps:
                if t%2==0:
                    new_theta = self.on_theta_update(theta, sigma_fun)
                    new_sigma_fun = sigma_fun
                else:
                    new_theta = theta
                    new_sigma_fun = self.on_sigma_update(theta, sigma_fun)
            else:
                new_theta = self.on_theta_update(theta, sigma_fun)
                new_sigma_fun = self.on_sigma_update(theta, sigma_fun)

            theta = new_theta
            sigma_fun = new_sigma_fun

            self.on_after_update(theta, sigma_fun)

            if theta >= 0:
                break


        if filename is not None:
            traj_data, columns = self.get_traj_data()

            traj_df = pd.DataFrame(traj_data, columns=columns)
            traj_df.to_pickle(filename + '.gzip')

            with open(filename + '_params.json', 'w') as f:
                json.dump({a:str(b) for a,b in initial_configuration.items()}, f)

    def __str__(self):
        if self.exp_name is None:
            return self.__class__.__name__
        else:
            return self.exp_name

class ExpLambda(Experiment):
    """This implements: σ <- σ + α[λ * ∇σJ + (1 -λ) * ∇σΔJ]
    """
    def __init__(self, lqg_environment, alphaSigma, lambda_coeff, exp_name = None):
        super().__init__(lqg_environment, exp_name)
        self.alphaSigma = alphaSigma
        self.lambda_coeff = lambda_coeff

    def on_sigma_update(self, theta, sigma_fun):
        """This implements: σ <- σ + α[λ * ∇σJ + (1 -λ) * ∇σΔJ]
        """
        sigma = sigma_fun.eval()
        updateGradSigma = self._get_gradient_sigma(theta, sigma)

        if isinstance(sigma_fun, Exponential):
            updateGradW = updateGradSigma * math.exp(sigma_fun.param)  # Computes the gradient wrt w
            return sigma_fun.update(self.alphaSigma, updateGradW)
        else:
            return sigma_fun.update(self.alphaSigma, updateGradSigma)

    def _get_gradient_sigma(self, theta, sigma):
        """Evaluates: λ * ∇σJ + (1 -λ) * ∇σΔJ
        """
        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        alphaStar = (self.C1 * sigma**3) / (self.m * (2 * self.C2 * sigma + self.C3))
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted

        updateGradSigma = self.lambda_coeff * self.gradSigma + (1 - self.lambda_coeff)*gradDelta

        return updateGradSigma

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['lambda_coeff'] = self.lambda_coeff
        ret_params['alphaSigma'] = self.alphaSigma

        return ret_params


class ExpSigmaTwoStep(Experiment):   # ULTRASAFE
    """This implements 2-step update using the gradient on theta and on sigma
    """
    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)  # The bound only works if sigma is exponential

        sigma = sigma_fun.eval()

        c_sigma = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        alpha_sigma = 1/(2 * c_sigma)

        return sigma_fun.update(alpha_sigma, self.gradW)

class ExpArgmaxExact(Experiment):
    """This experiment performs w <- w + b * ∇w J
        where w = argmax { Δ_w J(θ,w) + Δ_θ J(θ,w') }
    """
    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        MIN_STEP = -0.001
        MAX_STEP = 0.001
        STEP_COUNT = 100

        # scipy.optimize

        sigma = sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        a_star=1/(2*c)

        # Perform a linear search
        beta = np.linspace(MIN_STEP, MAX_STEP, STEP_COUNT)

        values = np.zeros(beta.shape[0])
        for t in range(values.shape[0]):
            b = beta[t]
            new_sigma = (sigma_fun.update(b, self.gradW)).eval() # == math.exp(sigma_fun.param + b*self.gradW)
            new_c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, new_sigma)
            a_star = 1/(2 * new_c)

            gradK_squared = utils.calc_K(theta, new_sigma, self.ENV_GAMMA, self.ENV_R, self.ENV_Q, self.M)**2

            term1 = (b - d * b**2)*self.gradW**2    # Δ_w J(θ,w)
            term2 = 0.5*a_star * gradK_squared      # Δ_θ J(θ,w')
            values[t] = term1 + term2

        idx = np.argmax(values)

        return sigma_fun.update(beta[idx], self.gradW)

class ExpArgmaxTaylor(Experiment):
    """This experiment performs w <- w + b * ∇w J
        where w is maximized with the First Order Taylor expansion of ExpArgmaxExact.
        In particular w = 1/(2*d) [1 + (∇w Δθ J) / (∇w J)]
    """
    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        alphaStar=1/(2*c)


        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        # alphaStar = (self.C1 * sigma**3) / (self.m * (2 * self.C2 * sigma + self.C3))
        # alphaStar = (self.C1 * sigma**3) / (self.m * (self.C2 * sigma + self.C3))

        # print("{} == {}".format(a_star, alphaStar))
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted


        grad_w_bound = gradDelta * math.exp(sigma_fun.param)
        #beta_star = 1/(2*d) * (1 + (self.gradMixedW) / (self.gradW))
        beta_star = 1/(2*d) * (1 + (grad_w_bound) / (self.gradW))
        #print("beta_star: {}, grad_w_bound/gradW: {}".format(beta_star, (grad_w_bound) / (self.gradW)))
        # input()

        return sigma_fun.update(beta_star, self.gradW)

class ExpSafeConstraint(Experiment):
    """This experiment optimizes a function with a safe constraint on the step size of w
    """
    def on_theta_update(self, theta, sigma_fun):
        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        alpha=1/(2*c)

        self.perf_improvement = (1/2) * alpha * self.gradK**2

        return theta + alpha*self.gradK



    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        a_star=1/(2*c)

        max_improvement = self.perf_improvement

        low = (1 - math.sqrt(1 - (4 * d * (-max_improvement))/(self.gradW**2))) / (2 * d)
        high = (1 + math.sqrt(1 - (4 * d * (-max_improvement))/(self.gradW**2))) / (2 * d)


        #grad = np.clip(gradDelta, low, high)
        step = low  # Fix the step to be the lowest one

        return sigma_fun.update(step, self.gradW)

class ExpSafeConstraintBaseline(Experiment):
    """This experiment optimizes a function with a safe constraint wrt a baseline on the step size of w
    """
    def __init__(self, lqg_environment, exp_name = None):
        super().__init__(lqg_environment, exp_name)
        self.is_first_time = True
        self.J_BASELINE = None

    def on_before_update(self, current_theta, current_sigma_fun):
        super().on_before_update(current_theta, current_sigma_fun)
        if self.is_first_time:
            self.J_BASELINE = self.J        # Set the baseline the first time the policy is evaluated
            self.is_first_time = False


    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        a_star=1/(2*c)

        max_improvement = self.J - self.J_BASELINE #self.perf_improvement

        step = (1 - math.sqrt(1 - (4 * d * (-max_improvement))/(self.gradW**2))) / (2 * d)

        return sigma_fun.update(step, self.gradW)

class ExpSafeConstraintBudget(Experiment):
    """Exploration cost is defined by a Budget, that is earned when improving theta, and lost when not following the gradient
    """
    def __init__(self, lqg_environment, exp_name = None, gamma_coeff=0.0, initial_budget = 0):
        super().__init__(lqg_environment, exp_name)
        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.budget_data = np.zeros(self.UPDATE_DIM)

        self.gamma_coeff = gamma_coeff

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['gamma_coeff'] = self.gamma_coeff
        ret_params['initial_budget'] = self.initial_budget

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget


    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data, self.budget_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET']

    def on_theta_update(self, theta, sigma_fun):
        new_theta = super().on_theta_update(theta, sigma_fun)

        sigma = sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        self.budget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        return new_theta

    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        a_star=1/(2*c)

        # assert that the budget is small enough
        if self.budget >= -(self.gradW**2)/(4*d):
            beta_tilde = (1 - math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)
        else:
            beta_tilde = 1/(2*d)
        #self.budget = max(self.budget, -(self.gradW**2)/(4*d))

        #beta_tilde = (1 - math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)

        new_sigma_fun = sigma_fun.update(beta_tilde, self.gradW)
        new_sigma = new_sigma_fun.eval()

        # Reduce the budget due to lost exploitation
        self.budget -= (1-math.pow(self.gamma_coeff, self.t))*((self.gradW**2) / (4 * d))

        # Improve the budget due to performance improvement
        self.budget += utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B) - self.J

        return new_sigma_fun


class ExpExplorationCost(Experiment):
    """Exploration cost is defined as J(theta, 0) - J(theta, sigma).
    Initial exploration cost is given, then it will be discounted by a factor lambda
    """
    def __init__(self, lqg_environment, exp_name = None, init_exp = -1, discount_coeff=1.0):
        """
        init_exp: Float or -1
            If set to a positive value, then this will define the initial exploration cost.
            If set to -1, then the initial exploration cost will be defined by the initial policy
        """
        super().__init__(lqg_environment, exp_name)
        self.init_exp = init_exp
        self.current_exp = None

        self.discount_coeff = discount_coeff

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['discount_coeff'] = self.discount_coeff
        ret_params['init_exp'] = self.init_exp

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        if self.current_exp is None:
            if self.init_exp == -1:
                self.current_exp = self.exploration_data[0]
            else:
                self.current_exp = self.init_exp

    def on_sigma_update(self, theta, sigma_fun):
        self.current_exp = self.current_exp * self.discount_coeff
        #self.current_exp = max(MIN_SIGMA, self.current_exp - self.discount_coeff)
        new_sigma = utils.calc_optimal_sigma(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, self.ENV_B, self.current_exp)

        if isinstance(sigma_fun, Exponential):
            w = np.clip(math.log(new_sigma), math.log(MIN_SIGMA), math.log(MAX_SIGMA))
            return Exponential(w)
        else:
            return Identity(new_sigma)


class ExplorationBudget(Experiment):
    def __init__(self, lqg_environment, exp_name = None, gamma_coeff=0.0, initial_budget = 0):
        super().__init__(lqg_environment, exp_name)
        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.budget_data = np.zeros(self.UPDATE_DIM)

        self.gamma_coeff = gamma_coeff

        self.first_time = True

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['gamma_coeff'] = self.gamma_coeff
        ret_params['initial_budget'] = self.initial_budget

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget


    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data, self.budget_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET']

    def on_theta_update(self, theta, sigma_fun):
        new_theta = super().on_theta_update(theta, sigma_fun)

        sigma = sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        alpha=1/(2*c)

        grad_det = utils.calc_K(theta, 0, self.ENV_GAMMA, self.ENV_R, self.ENV_Q, self.M)
        self.budget_increment = alpha * (self.gradK)**2 - alpha * (grad_det)**2

        #self.budget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J
        # self.budget += alpha * (self.gradK)**2
        #
        # grad_det = utils.calc_K(theta, 0, self.ENV_GAMMA, self.ENV_R, self.ENV_Q, self.M)
        # print(grad_det)
        # self.budget -= alpha * (grad_det)**2


        return new_theta

    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)
        sigma = sigma_fun.eval()
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        if self.first_time:
            self.budget = (self.gradW**2)/(4*d)

            self.first_time = False



        a_star=1/(2*c)

        max_improvement = (self.gradW**2)/(4*d) - self.budget

        # assert that the budget is small enough

        beta_tilde = (1 - math.sqrt(1 - (4 * d * (max_improvement))/(self.gradW**2))) / (2 * d)


        new_sigma_fun = sigma_fun.update(beta_tilde, self.gradW)
        new_sigma = new_sigma_fun.eval()

        # Reduce the budget due to lost exploitation
        #self.budget -= (1-self.gamma_coeff)*((self.gradW**2) / (4 * d))

        # Improve the budget due to performance improvement
        #self.budget -= (self.J - utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B))
        self.budget += beta_tilde * (self.gradW)**2 + self.budget_increment

        return new_sigma_fun


class ExpDeterministicPolicy(Experiment):
    """Evaluate deterministic policy
    """
    def __init__(self, lqg_environment, exp_name = None, gamma_coeff=0.0, initial_budget = 0):
        super().__init__(lqg_environment, exp_name)
        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.budget_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

        self.gamma_coeff = gamma_coeff

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['gamma_coeff'] = self.gamma_coeff
        ret_params['initial_budget'] = self.initial_budget

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                    self.budget_data[:self.num_updates].reshape(-1,1),
                    self.detJ_data[:self.num_updates].reshape(-1,1),
                    self.JplusDetJ_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        new_theta = super().on_theta_update(theta, sigma_fun)

        sigma = sigma_fun.eval()
        self.budget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget += J_det - prev_det


        return new_theta

    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        alphaStar=1/(2*c)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)

        # assert that the budget is small enough
        if self.budget >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW



        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement
        self.budget += utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B) - self.J

        return new_sigma_fun


class ExpDiscountedDeterministicPolicy(Experiment):
    """Evaluate deterministic policy
    """
    def __init__(self, lqg_environment, exp_name = None, gamma_coeff=0.0, initial_budget = 0):
        super().__init__(lqg_environment, exp_name)
        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.budget_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

        self.gamma_coeff = gamma_coeff

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['gamma_coeff'] = self.gamma_coeff
        ret_params['initial_budget'] = self.initial_budget

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                    self.budget_data[:self.num_updates].reshape(-1,1),
                    self.detJ_data[:self.num_updates].reshape(-1,1),
                    self.JplusDetJ_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        new_theta = super().on_theta_update(theta, sigma_fun)

        sigma = sigma_fun.eval()
        self.deltaBudget = 0
        self.deltaBudget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.deltaBudget += J_det - prev_det


        return new_theta

    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        alphaStar=1/(2*c)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)

        # assert that the budget is small enough
        if self.budget >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW



        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement
        self.deltaBudget += utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B) - self.J

        cost = (utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B) - self.J)

        self.budget += (math.pow(self.gamma_coeff, self.t)*self.deltaBudget - (1 - self.gamma_coeff) * (cost)) / (math.pow(self.gamma_coeff, self.t) + 1 - self.gamma_coeff)

        return new_sigma_fun






class ExpGuaranteedOnlineDeterministicPolicy(ExpDeterministicPolicy):
    """Evaluate deterministic policy
    """

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        sigma = current_sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        #max_budget = max(self.budget, self.budget - 0.9*(self.gradW**2)/(4*d))


        #self.extra_budget = self.budget - min(self.budget, max_budget)
        #self.budget = self.budget - self.extra_budget
        self.budget -= self.gamma_coeff*(self.gradW**2)/(4*d)
        #self.budget -= (1 - math.pow(0.999999, self.t))*(self.gradW**2)/(4*d)

    def on_after_update(self, current_theta, current_sigma_fun):
        #self.budget += self.extra_budget
        #self.extra_budget = 0
        super().on_after_update(current_theta, current_sigma_fun)



class ExpDiscountedGuaranteedOnlineDeterministicPolicy(ExpDeterministicPolicy):
    """Evaluate deterministic policy
    """

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        sigma = current_sigma_fun.eval()
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        #max_budget = max(self.budget, self.budget - 0.9*(self.gradW**2)/(4*d))


        #self.extra_budget = self.budget - min(self.budget, max_budget)
        #self.budget = self.budget - self.extra_budget
        self.budget -= (1 - math.pow(self.gamma_coeff, self.t))*(self.gradW**2)/(4*d)
        #self.budget -= (1 - math.pow(0.999999, self.t))*(self.gradW**2)/(4*d)

    def on_after_update(self, current_theta, current_sigma_fun):
        #self.budget += self.extra_budget
        #self.extra_budget = 0
        super().on_after_update(current_theta, current_sigma_fun)




class ExpEarlyStoppingCondition(ExpDeterministicPolicy):
    """Stop condition: J(theta, 0) >= (1-gamma)*J(theta, sigma) + gamma*J(theta', 0)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.STOP_UPDATE = False

    def on_theta_update(self, theta, sigma_fun):
        if self.STOP_UPDATE:
            return 0

        prev_theta = theta

        new_theta = super().on_theta_update(theta, sigma_fun)

        J_det_old = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        J_det_new = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        if (J_det_old) >= ((1 - self.gamma_coeff)*self.J + self.gamma_coeff * (J_det_new)):
            self.STOP_UPDATE = True

        return new_theta

    def on_sigma_update(self, theta, sigma_fun):
        if self.STOP_UPDATE:
            return sigma_fun
        else:
            return super().on_sigma_update(theta, sigma_fun)


class ExpBudgetReductionStopCondition(ExpDeterministicPolicy):
    """Budget condition: J(theta, sigma) >= J(theta, 0)/(1-gamma) - gamma/(1-gamma)*J(theta', 0)
    """
    def on_theta_update(self, theta, sigma_fun):
        prev_theta = theta

        new_theta = super().on_theta_update(theta, sigma_fun)

        J_det_old = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        J_det_new = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget = min(self.budget, -((J_det_old - self.gamma_coeff*J_det_new)/(1 - self.gamma_coeff) - self.J))

        return new_theta





class ExpThetaAndSigmaBudget(Experiment):
    def __init__(self, lqg_environment, exp_name=None, alpha_coeff=0.0, beta_coeff=0.0, initial_budget=0):
        super().__init__(lqg_environment, exp_name)
        self.budget = initial_budget
        self.initial_budget = initial_budget

        self.budget_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)
        ret_params['alpha_coeff'] = self.alpha_coeff
        ret_params['beta_coeff'] = self.beta_coeff
        ret_params['initial_budget'] = self.initial_budget

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                self.budget_data[:self.num_updates].reshape(-1, 1),
                self.detJ_data[:self.num_updates].reshape(-1, 1),
                self.JplusDetJ_data[:self.num_updates].reshape(-1, 1)]), columns + ['BUDGET', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        self.budget -= self.alpha_coeff*(self.gradK**2)/(4*c)

        budget_theta = self.beta_coeff * self.budget

        if budget_theta >= -(self.gradK**2)/(4*c):
            alpha_star = (1 + math.sqrt(1 - (4 * c * (-budget_theta))/(self.gradK**2))) / (2 * c)
        else:
            alpha_star = 1/(2*c)

        new_theta = theta + alpha_star*self.gradK

        self.budget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget += J_det - prev_det


        return new_theta


    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        self.budget -= self.alpha_coeff*(self.gradW**2)/(4*d)

        alphaStar = 1/(2*c)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)

        budgetW = (1 - self.alpha_coeff - self.beta_coeff)*self.budget

        # assert that the budget is small enough
        if budgetW >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-budgetW))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-budgetW))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW



        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement
        self.budget += utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B) - self.J



        return new_sigma_fun



class ExpTwoBudgetsDeltaDelta(Experiment):
    def __init__(self, lqg_environment, exp_name=None):
        super().__init__(lqg_environment, exp_name)
        self.budget_theta = 0
        self.budget_sigma = 0

        self.prevDelta = 0

        self.budget_theta_data = np.zeros(self.UPDATE_DIM)
        self.budget_sigma_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_theta_data.shape[0]:
                self.budget_theta_data = np.concatenate([self.budget_theta_data, np.zeros(self.UPDATE_DIM)])
                self.budget_sigma_data = np.concatenate([self.budget_sigma_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_theta_data[self.num_updates] = self.budget_theta
            self.budget_sigma_data[self.num_updates] = self.budget_sigma
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                    self.budget_theta_data[:self.num_updates].reshape(-1, 1),
                    self.budget_sigma_data[:self.num_updates].reshape(-1, 1),
                    self.detJ_data[:self.num_updates].reshape(-1, 1),
                    self.JplusDetJ_data[:self.num_updates].reshape(-1, 1)]), columns + ['BUDGET_THETA', 'BUDGET_SIGMA', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        sigma = sigma_fun.eval()
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)




        if self.budget_theta >= -(self.gradK**2)/(4*c):
            alpha_star = (1 + math.sqrt(1 - (4 * c * (-self.budget_theta))/(self.gradK**2))) / (2 * c)
        else:
            alpha_star = 1/(2*c)

        new_theta = theta + alpha_star*self.gradK

        # if self.t > 1:
        #     self.budget_theta += (utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J)
        # else:
        #     self.budget_sigma += (utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J)

        self.budget_theta += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget_theta += J_det - prev_det


        return new_theta


    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        # Transfer residual budget from theta to sigma
        if self.budget_theta > 0:
            self.budget_sigma += self.budget_theta
            self.budget_theta = 0

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        alphaStar=1/(2*c)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)



        # assert that the budget is small enough
        if self.budget_sigma >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget_sigma))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget_sigma))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW



        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement


        new_j = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B)
        deltaPerf = new_j - self.J

        self.budget_sigma += deltaPerf

        # print(new_j, self.J, deltaPerf, self.prevDelta, (deltaPerf - self.prevDelta))

        # coeff = math.pow(0.999, self.t) / (1 - 0.999)
        # self.budget_sigma += coeff*(deltaPerf - self.prevDelta)
        # self.budget_theta -= coeff*(deltaPerf - self.prevDelta)

        # self.budget_sigma += (deltaPerf - self.prevDelta)
        # self.budget_theta -= (deltaPerf - self.prevDelta)
        # self.prevDelta = deltaPerf



        return new_sigma_fun



class ExpAlternatingBudget(Experiment):
    def __init__(self, lqg_environment, exp_name = None):
        super().__init__(lqg_environment, exp_name)
        self.budget_theta = 0
        self.budget_sigma = 0

        self.prevDelta = 0

        self.budget_theta_data = np.zeros(self.UPDATE_DIM)
        self.budget_sigma_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

    def get_param_list(self, params):
        ret_params = super().get_param_list(params)

        return ret_params

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_theta_data.shape[0]:
                self.budget_theta_data = np.concatenate([self.budget_theta_data, np.zeros(self.UPDATE_DIM)])
                self.budget_sigma_data = np.concatenate([self.budget_sigma_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_theta_data[self.num_updates] = self.budget_theta
            self.budget_sigma_data[self.num_updates] = self.budget_sigma
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                    self.budget_theta_data[:self.num_updates].reshape(-1,1),
                    self.budget_sigma_data[:self.num_updates].reshape(-1,1),
                    self.detJ_data[:self.num_updates].reshape(-1,1),
                    self.JplusDetJ_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET_THETA', 'BUDGET_SIGMA', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        sigma = sigma_fun.eval()
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        # Transfer residual budget of sigma to theta
        if self.budget_sigma > 0:
            self.budget_theta += self.budget_sigma
            self.budget_sigma = 0


        if self.budget_theta >= -(self.gradK**2)/(4*c):
            alpha_star = (1 + math.sqrt(1 - (4 * c * (-self.budget_theta))/(self.gradK**2))) / (2 * c)
        else:
            alpha_star = 1/(2*c)

        new_theta = theta + alpha_star*self.gradK

        # if self.t > 1:
        #     self.budget_theta += (utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J)
        # else:
        #     self.budget_sigma += (utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J)

        self.budget_theta += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget_theta += J_det - prev_det


        return new_theta


    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        # Transfer residual budget from theta to sigma
        if self.budget_theta > 0:
            self.budget_sigma += self.budget_theta
            self.budget_theta = 0

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        alphaStar=1/(2*c)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)



        # assert that the budget is small enough
        if self.budget_sigma >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget_sigma))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget_sigma))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW



        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement


        new_j = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B)
        deltaPerf = new_j - self.J

        self.budget_sigma += deltaPerf

        # print(new_j, self.J, deltaPerf, self.prevDelta, (deltaPerf - self.prevDelta))

        # coeff = math.pow(0.999, self.t) / (1 - 0.999)
        # self.budget_sigma += coeff*(deltaPerf - self.prevDelta)
        # self.budget_theta -= coeff*(deltaPerf - self.prevDelta)

        # self.budget_sigma += (deltaPerf - self.prevDelta)
        # self.budget_theta -= (deltaPerf - self.prevDelta)
        # self.prevDelta = deltaPerf



        return new_sigma_fun




class ExpSingleBudget(Experiment):
    def __init__(self, lqg_environment, exp_name=None):
        super().__init__(lqg_environment, exp_name)
        self.budget = 0

        self.prevDelta = 0

        self.budget_data = np.zeros(self.UPDATE_DIM)
        self.detJ_data = np.zeros(self.UPDATE_DIM)
        self.JplusDetJ_data = np.zeros(self.UPDATE_DIM)

    def on_before_update(self, current_theta, current_sigma_fun):
        """Store budget
        """
        super().on_before_update(current_theta, current_sigma_fun)

        # Save budget data
        if self.t % self.downsample == 0:
            if self.num_updates >= self.budget_data.shape[0]:
                self.budget_data = np.concatenate([self.budget_data, np.zeros(self.UPDATE_DIM)])
                self.detJ_data = np.concatenate([self.detJ_data, np.zeros(self.UPDATE_DIM)])
                self.JplusDetJ_data = np.concatenate([self.JplusDetJ_data, np.zeros(self.UPDATE_DIM)])

            self.budget_data[self.num_updates] = self.budget
            self.detJ_data[self.num_updates] = utils.calc_J(current_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
            self.JplusDetJ_data[self.num_updates] = self.J + self.detJ_data[self.num_updates]



    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        data, columns = super().get_traj_data()
        return np.hstack([data,
                    self.budget_data[:self.num_updates].reshape(-1,1),
                    self.detJ_data[:self.num_updates].reshape(-1,1),
                    self.JplusDetJ_data[:self.num_updates].reshape(-1,1)]), columns + ['BUDGET', 'J_DET', 'J+J_DET']

    def on_theta_update(self, theta, sigma_fun):
        """Perform a safe update on theta
        """
        sigma = sigma_fun.eval()
        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)


        if self.budget >= -(self.gradK**2)/(4*c):
            alpha_star = (1 + math.sqrt(1 - (4 * c * (-self.budget))/(self.gradK**2))) / (2 * c)
        else:
            alpha_star = 1/(2*c)

        new_theta = theta + alpha_star*self.gradK

        self.budget += utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, sigma, self.M, self.ENV_B) - self.J

        J_det = utils.calc_J(new_theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)
        prev_det = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, 0, self.M, self.ENV_B)

        self.budget += J_det - prev_det


        return new_theta


    def on_sigma_update(self, theta, sigma_fun):
        assert isinstance(sigma_fun, Exponential)

        sigma = sigma_fun.eval()

        c = utils.computeLoss(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)
        d = utils.computeLossSigma(self.MAX_REWARD, self.M, self.ENV_GAMMA, self.ENV_VOLUME, sigma)

        print('gradK', self.gradK, 'gradMixed', self.gradMixed, 'gradW', self.gradW)

        print('c:', c, 'd: ', d)
        alphaStar=1/(2*c)

        print('alphaStar:', alphaStar)

        grad_sigma_alpha_star = sigma**2 * (2*self.C1*self.C2*sigma + 3*self.C1*self.C3) / (self.m * (self.C2 * sigma + self.C3)**2)
        grad_sigma_norm_grad_theta = 2 * self.gradK * self.gradMixed

        print('grad_sigma_alpha_star', grad_sigma_alpha_star, 'grad_sigma_norm_grad_theta: ', grad_sigma_norm_grad_theta)

        # Compute the gradient for sigma
        grad_local_step = (1/2) * self.gradK**2 * grad_sigma_alpha_star
        grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

        print('grad_local_step', grad_local_step, 'grad_far_sighted', grad_far_sighted)

        gradDelta = grad_local_step + grad_far_sighted
        gradDeltaW = gradDelta * math.exp(sigma_fun.param)

        print('gradDelta', gradDelta, 'gradDeltaW', gradDeltaW)



        # assert that the budget is small enough
        if self.budget >= -(self.gradW**2)/(4*d):
            beta_tilde_minus = (1 - math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)
            beta_tilde_plus = (1 + math.sqrt(1 - (4 * d * (-self.budget))/(self.gradW**2))) / (2 * d)

            if gradDeltaW / self.gradW >= 0:
                beta_star = beta_tilde_plus * self.gradW / gradDeltaW
            else:
                beta_star = beta_tilde_minus * self.gradW / gradDeltaW

        else:
            beta_star = 1/(2*d) * self.gradW / gradDeltaW

        print('beta_star', beta_star)
        new_sigma_fun = sigma_fun.update(beta_star, gradDeltaW)
        new_sigma = new_sigma_fun.eval()

        # Improve the budget due to performance improvement

        new_j = utils.calc_J(theta, self.ENV_Q, self.ENV_R, self.ENV_GAMMA, new_sigma, self.M, self.ENV_B)
        deltaPerf = new_j - self.J

        self.budget += deltaPerf



        time.sleep(10000)

        return new_sigma_fun


class ExpDiscountedSingleBudget(ExpSingleBudget):
    def __init__(self, lqg_environment, exp_name=None, discount_coeff=1.0):
        super().__init__(lqg_environment, exp_name=exp_name)
        self.discount_coeff = discount_coeff

    def get_param_list(self, params):
        new_params = super().get_param_list(params)
        new_params['discount_coeff'] = self.discount_coeff

        return new_params

    def on_before_update(self, current_theta, current_sigma_fun):
        self.budget *= self.discount_coeff
        super().on_before_update(current_theta, current_sigma_fun)

class ExpSingleBudgetRandomBaselines(ExpSingleBudget):
    def __init__(self, lqg_environment, exp_name=None, baseline_prob=0.0):
        super().__init__(lqg_environment, exp_name=exp_name)
        self.baseline_prob = baseline_prob

    def get_param_list(self, params):
        new_params = super().get_param_list(params)
        new_params['baseline_prob'] = self.baseline_prob

        return new_params

    def on_before_update(self, current_theta, current_sigma_fun):
        if np.random.rand() >= self.baseline_prob:
            self.budget = 0

        super().on_before_update(current_theta, current_sigma_fun)


def generate_filename():
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))











if __name__ == '__main__':
    BASE_FOLDER = 'experiments_long'
    maybe_make_dir(BASE_FOLDER)

    N_ITERATIONS = -1
    EPS = 1e-03
    INIT_THETA = -0.1

    #gamma_coeffs = [0, 0.2, 0.4, 0.6, 0.8, 0.99, 0.99999, 1]
    #gamma_coeffs = [1-1e-12, 1-1e-13, 1-1e-14]
    # baselines_probs = [0.01, 0.05, 0.1, 0.2, 0.5]
    # for bprob in baselines_probs:
    #     l = Exponential(0)
    #     e = ExpSingleBudgetRandomBaselines(LQG_ENV, "ExpSingleBudgetRandomBaselines", baseline_prob=bprob)
    #     filename = os.path.join(BASE_FOLDER, generate_filename())
    #     e.run(theta=INIT_THETA, sigma_fun=l, eps=EPS, verbose=False, two_steps=True, n_iterations=N_ITERATIONS, filename=filename, downsample=5)
    #
    # exit()


    l = Exponential(math.log(1))
    e = ExpSingleBudget(LQG_ENV, "ExpSingleBudget")
    filename = os.path.join(BASE_FOLDER, generate_filename())
    e.run(theta=-0.1, sigma_fun=l, eps=EPS, verbose=False, two_steps=True, n_iterations=N_ITERATIONS, filename=None, downsample=5)

    exit()

    # for alpha_coeff in [0]:
    #     for beta_coeff in [1]:
    #         if alpha_coeff + beta_coeff > 1:
    #             continue
    #         l = Exponential(math.log(1.5))
    #         e = ExpThetaAndSigmaBudget(LQG_ENV, "ExpThetaAndSigmaBudget", alpha_coeff=alpha_coeff, beta_coeff=beta_coeff)
    #         filename = os.path.join(BASE_FOLDER, generate_filename())
    #         e.run(theta=INIT_THETA, sigma_fun=l, eps=EPS, verbose=True, two_steps=True, n_iterations=N_ITERATIONS, filename=filename, downsample=1)
    #
    # exit()

    #
    # l = Exponential(0)
    # e = ExplorationBudget(LQG_ENV, "ExplorationBudget_Taylor6")
    # filename = os.path.join(BASE_FOLDER, generate_filename())
    # e.run(theta=INIT_THETA, sigma_fun=l, eps=EPS, verbose=True, two_steps=False, n_iterations=N_ITERATIONS, filename=filename, downsample=1)
    #
    # exit()
    #
    # l = Exponential(0)
    # e = ExpExplorationCost(LQG_ENV, "ExpExplorationCost", discount_coeff=1.0, init_exp=-1)
    # filename = os.path.join(BASE_FOLDER, generate_filename())
    # e.run(theta=INIT_THETA, sigma_fun=l, eps=EPS, verbose=True, two_steps=True, n_iterations=N_ITERATIONS, filename=filename, downsample=1)
    #
    # exit()

    # for discount_coeff in [1, 0.1, 0.01, 0.001, 0.0001,0.0005, 0.00001]:
    #     for init_exp in [-1]:
    #         l = Exponential(0)
    #         e = ExpExplorationCost(LQG_ENV, "ExpExplorationCostConstant_two_steps", discount_coeff=discount_coeff, init_exp=init_exp)
    #         filename = os.path.join(BASE_FOLDER, generate_filename())
    #         e.run(theta=INIT_THETA, sigma_fun=l, eps=EPS, verbose=False, two_steps=True, n_iterations=N_ITERATIONS, filename=filename, downsample=10)
    #
    # exit()


    # gamma_coeffs = [0.999, 0.9999, 0.99999] #gamma_coeffs = [0, 0.2, 0.5, 0.8, 0.99, 1]
    # initial_budget = [0, 100, 200]
    # for gamma_coeff in gamma_coeffs:
    #     for init_budget in initial_budget:
    #         e = ExpSafeConstraintBudgetFork(LQG_ENV, "ExpSafeConstraintBudgetFork_two_steps", gamma_coeff=gamma_coeff, initial_budget=init_budget)
    #         filename = os.path.join(BASE_FOLDER, generate_filename())
    #         print("Running experiment: ExpSafeConstraintBudgetFork_two_steps(gamma_coeff={})".format(gamma_coeff))
    #         e.run(  theta=INIT_THETA,
    #                 sigma_fun=Exponential(0),
    #                 eps=EPS,
    #                 verbose=False,
    #                 two_steps=True,
    #                 n_iterations=N_ITERATIONS,
    #                 filename=filename,
    #                 downsample=4)
    #
    #
    # exit()



    #
    #   RUN ALL STANDALONE EXPERIMENTS
    #

    experiments = [ExpSigmaTwoStep, ExpSafeConstraint, ExpSafeConstraintBaseline, ExpArgmaxTaylor]
    exp_names = ["ExpSigmaTwoStep", "ExpSafeConstraint", "ExpSafeConstraintBaseline", "ExpArgmaxTaylor"]

    for exp_class,name in zip(experiments, exp_names):
        for v,count_step in zip([False, True], ["one_step", "two_steps"]):
            print("Running experiment: {}_{}".format(name, count_step))
            filename = os.path.join(BASE_FOLDER, generate_filename())

            exp = exp_class(LQG_ENV, exp_name="%s_%s" % (name, count_step))

            exp.run(    theta=INIT_THETA,
                        sigma_fun=Exponential(0),
                        n_iterations=N_ITERATIONS,
                        eps=EPS,
                        filename=filename,
                        verbose=False,
                        two_steps=v)

    #
    #   RUN LAMBDA EXPERIMENTS
    #


    lambda_experiments = [ExpLambda(LQG_ENV, alphaSigma, lambda_coeff, exp_name='Lambda_one_step') \
                    for alphaSigma in [0, 0.005, 0.01, 0.001, 0.0005, 0.02] \
                    for lambda_coeff in [0, 0.000001, 0.00001, 0.000005, 1]]

    for exp in lambda_experiments:
        filename = os.path.join(BASE_FOLDER, generate_filename())
        print("Running experiment: Lambda(lambda={}, alpha_sigma={})".format(exp.lambda_coeff, exp.alphaSigma))
        exp.run(theta=INIT_THETA,
                sigma_fun=Exponential(0),
                n_iterations=N_ITERATIONS,
                eps=EPS,
                filename=filename,
                verbose=False,
                two_steps=False)

    #
    #   RUN BUDGET EXPERIMENTS
    #
    gamma_coeffs = [0, 0.2, 0.5, 0.8, 0.99, 1]

    for gamma_coeff in gamma_coeffs:
        e = ExpSafeConstraintBudget(LQG_ENV, "ExpSafeConstraintBudget_two_steps", gamma_coeff=gamma_coeff)
        filename = os.path.join(BASE_FOLDER, generate_filename())
        print("Running experiment: ExpSafeConstraintBudget_two_steps(gamma_coeff={})".format(gamma_coeff))
        e.run(theta=INIT_THETA, sigma_fun=Exponential(0), eps=EPS, verbose=False, two_steps=True, n_iterations=N_ITERATIONS, filename=filename)
