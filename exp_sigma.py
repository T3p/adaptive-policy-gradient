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

try:
    import numba
    NUMBA_PRESENT = True
except ImportError:
    NUMBA_PRESENT = False


MIN_SIGMA = 0.1
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

        self.J_data = np.zeros(self.UPDATE_DIM)
        self.theta_data = np.zeros(self.UPDATE_DIM)
        self.sigma_data = np.zeros(self.UPDATE_DIM)
        self.grad_K_data = np.zeros(self.UPDATE_DIM)
        self.gradSigma_data = np.zeros(self.UPDATE_DIM)
        self.gradMixed_data = np.zeros(self.UPDATE_DIM)

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
        self.C3 = 2*(1 - self.ENV_GAMMA) * self.ENV_MAX_ACTION * self.MAX_REWARD * self.M**2

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

        if self.t >= self.num_rows:
            self.J_data = np.concatenate([self.J_data, np.zeros(self.UPDATE_DIM)])
            self.theta_data = np.concatenate([self.theta_data, np.zeros(self.UPDATE_DIM)])
            self.sigma_data = np.concatenate([self.sigma_data, np.zeros(self.UPDATE_DIM)])
            self.grad_K_data = np.concatenate([self.grad_K_data, np.zeros(self.UPDATE_DIM)])
            self.gradSigma_data = np.concatenate([self.gradSigma_data, np.zeros(self.UPDATE_DIM)])
            self.gradMixed_data = np.concatenate([self.gradMixed_data, np.zeros(self.UPDATE_DIM)])

            self.num_rows += self.UPDATE_DIM

        self.J_data[self.t] = self.J
        self.theta_data[self.t] = current_theta
        self.sigma_data[self.t] = current_sigma
        self.grad_K_data[self.t] = self.gradK
        self.gradSigma_data[self.t] = self.gradSigma
        self.gradMixed_data[self.t] = self.gradMixed

    def on_after_update(self, current_theta, current_sigma_fun):
        self.t += 1


    def get_traj_data(self):
        """Returns a matrix containing all the data we want to store
        """
        return np.hstack([  np.arange(0, self.num_rows).reshape(-1,1),
                            self.J_data.reshape(-1,1),
                            self.theta_data.reshape(-1,1),
                            self.sigma_data.reshape(-1,1),
                            self.grad_K_data.reshape(-1,1),
                            self.gradSigma_data.reshape(-1,1),
                            self.gradMixed_data.reshape(-1,1)])


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
            two_steps=True):
        """Runs the experiment.
        """
        initial_configuration = self.get_param_list(locals())

        for t in range_unlimited(n_iterations):
            # Early stopping if converged or diverged
            if abs(OPTIMAL_K - theta) <= eps:
                break
            if abs(theta) > 10:
                print("DIVERGED")
                break

            sigma = sigma_fun.eval()
            self.on_before_update(theta, sigma_fun)

            # Display
            if verbose:
                if t % 100 == 0:
                    print("\nT\t\tTheta\t\tSigma\t\tJ\n")
                print("{}\t\t{:.5f}\t\t{:.4f}\t\tJ{:.5f}".format(t, theta, sigma, self.J))

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


        if filename is not None:
            traj_data = self.get_traj_data()
            np.save(filename, traj_data[:t])
            with open(filename[:-4] + '_params.json', 'w') as f:
                json.dump({a:str(b) for a,b in initial_configuration.items()}, f)
            #np.save(filename[:-4] + '_params.npy', initial_configuration)

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
        step = low

        return sigma_fun.update(step, self.gradW)

def generate_filename():
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))


if __name__ == '__main__':
    # l = Identity(1)
    # e = LambdaExperiment(LQG_ENV, 0.0005, 1e-05)
    # e.run(sigma_fun=l, verbose=True, two_steps=False, n_iterations=100000)
    # #run_experiment(sigma_fun=l, theta=-0.1, alphaSigma=0.00002, lambda_coeff=0.00005, n_iterations=100000, verbose=False, two_steps=False)
    # exit()

    BASE_FOLDER = 'experiments_new'
    maybe_make_dir(BASE_FOLDER)

    N_ITERATIONS = 200000
    EPS = 1e-03
    INIT_THETA = -0.1


    #
    #   RUN ALL STANDALONE EXPERIMENTS
    #

    experiments = [ExpSigmaTwoStep, ExpArgmaxExact, ExpSafeConstraint]
    exp_names = ["ExpSigmaTwoStep", "ExpArgmaxExact", "ExpSafeConstraint"]

    for exp_class,name in zip(experiments, exp_names):
        for v,count_step in zip([False, True], ["one_step", "two_steps"]):
            print("Running experiment: {}_{}".format(name, count_step))
            filename = os.path.join(BASE_FOLDER, generate_filename()) + '.npy'

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


    lambda_experiments = [ExpLambda(LQG_ENV, alphaSigma, lambda_coeff) \
                    for alphaSigma in [0, 0.005, 0.01, 0.001, 0.0005, 0.02] \
                    for lambda_coeff in [0, 0.000001, 0.00001, 0.000005, 1]]

    for exp in lambda_experiments:
        filename = os.path.join(BASE_FOLDER, generate_filename()) + '.npy'
        print("Running experiment: Lambda(lambda={}, alpha_sigma={})".format(exp.lambda_coeff, exp.alphaSigma))
        exp.run(theta=INIT_THETA,
                sigma_fun=Exponential(0),
                n_iterations=N_ITERATIONS,
                eps=EPS,
                filename=filename,
                verbose=False,
                two_steps=False)
