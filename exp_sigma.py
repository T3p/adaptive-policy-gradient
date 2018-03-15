from enum import Enum
import lqg1d
import numpy as np
import math
import itertools
import os
import utils
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


lqg_env = lqg1d.LQG1D()

#
# Compute constants
#
R = np.asscalar(lqg_env.Q*lqg_env.max_pos**2 + lqg_env.R*lqg_env.max_action**2)
M = lqg_env.max_pos
gamma = lqg_env.gamma
volume = 2*lqg_env.max_action

ENV_R = np.asscalar(lqg_env.R)
ENV_Q = np.asscalar(lqg_env.Q)
ENV_B = np.asscalar(lqg_env.B)

c1 = (1 - lqg_env.gamma)**3 * math.sqrt(2 * math.pi)
c2 = lqg_env.gamma * math.sqrt(2 * math.pi) * R * M**2
c3 = 2*(1 - lqg_env.gamma) * lqg_env.max_action * R * M**2

m = 1


class SigmaFunction(object):
    def __init__(self, param):
        self.initial_param = param
        self.param = param
        self.description = 'sigma'

    def update(self, step_size, gradient_sigma):
        raise NotImplemented('Function not implemented')

    def eval(self):
        raise NotImplemented('Function not implemented')

    def reset(self):
        self.param = self.initial_param

    def __str__(self):
        return self.description


class Identity(SigmaFunction):
    def __init__(self, param):
        super().__init__(param)
        self.description = str(param)

    def update(self, step_size, gradient_sigma):
        new_param = self.param + step_size*gradient_sigma
        new_param = max(new_param, MIN_SIGMA)

        self.param = new_param

    def eval(self):
        return self.param

class Fixed(SigmaFunction):
    def __init__(self, param):
        super().__init__(param)
        self.description = 'F' + str(param)

    def update(self, step_size, gradient_sigma):
        pass

    def eval(self):
        return self.param

class Exponential(SigmaFunction):
    def __init__(self, param):
        super().__init__(param)
        self.description = 'e(' + str(self.param) + ')'

    def update(self, step_size, gradient_sigma):
        new_param = self.param + step_size * gradient_sigma * math.exp(self.param)
        new_param = min(new_param, math.log(MAX_SIGMA))
        new_param = max(new_param, math.log(MIN_SIGMA))

        self.param = new_param

    def eval(self):
        return math.exp(self.param)


def computeLoss(R, M, gamma, volume, sigma):
    return float(R*M**2)/((1-gamma)**2*sigma**2)* \
        (float(volume)/math.sqrt(2*math.pi*sigma**2) + \
            float(gamma)/(2*(1-gamma)))


def computeLossSigma(R, M, gamma, volume, sigma):
    c = (4*(math.sqrt(7) - 2)*math.exp((math.sqrt(7))/(2) - 2)) / (math.sqrt(2*math.pi))
    return R/((1-gamma)**2 *sigma) * ((c*volume) / (2) + (gamma) / ((1-gamma)*sigma))


def get_gradient_sigma(theta, sigma, lambda_coeff, gamma, R, Q, max_pos):
    gradK = utils.calc_K(theta, sigma, gamma, R, Q, max_pos)
    #gradSigma = lqg_env.grad_Sigma(theta, sigma)
    gradSigma = utils.calc_sigma(theta, Q, R, gamma)

    #gradMixed = lqg_env.grad_mixed(theta, sigma)
    gradMixed = utils.calc_mixed(gamma, theta, R, Q)

    grad_sigma_alpha_star = sigma**2 * (2*c1*c2*sigma + 3*c1*c3) / (m * (c2 * sigma + c3)**2)
    alphaStar = (c1 * sigma**3) / (m * (2 * c2 * sigma + c3))
    grad_sigma_norm_grad_theta = 2 * gradK * gradMixed

    # Compute the gradient for sigma
    grad_local_step = (1/2) * gradK**2 * grad_sigma_alpha_star
    grad_far_sighted = (1/2) * alphaStar * grad_sigma_norm_grad_theta

    gradDelta = grad_local_step + grad_far_sighted

    updateGradSigma = lambda_coeff * gradSigma + (1 - lambda_coeff)*gradDelta

    return updateGradSigma,gradK,gradSigma,gradMixed





#
#    COMPILE FUNCTIONS IF NUMBA IS PRESENT
#
if NUMBA_PRESENT:
    computeLoss = numba.jit(computeLoss)
    computeLossSigma = numba.jit(computeLossSigma)
    get_gradient_sigma = numba.jit(get_gradient_sigma)




def run_experiment( lambda_coeff = 0.1,
                    theta=-0.1,
                    sigma_fun=Identity(1),
                    alphaSigma=0.0001,
                    n_iterations=-1,
                    eps=1e-01,
                    filename=None,
                    verbose=True,
                    two_steps=False):
    initial_configuration = np.array([lambda_coeff, theta, str(sigma_fun), alphaSigma, n_iterations, eps, filename])


    sigma_fun.reset()
    sigma = sigma_fun.eval()

    traj_data = np.zeros((max(n_iterations, 50000), 9))

    for t in range_unlimited(n_iterations):
        sigma = sigma_fun.eval()

        J = utils.calc_J(theta, ENV_Q, ENV_R, lqg_env.gamma, sigma, lqg_env.max_pos, ENV_B)

        # Get the gradients
        updateGradSigma, gradK, gradSigma, gradMixed = get_gradient_sigma(theta, sigma, lambda_coeff, lqg_env.gamma, ENV_R, ENV_Q, lqg_env.max_pos)

        c = computeLoss(R, M, gamma, volume, sigma)
        alpha=1/(2*c)

        # Early stopping if converged or diverged
        if abs(gradK) <= eps:
            break
        if abs(theta) > 100:
            print("DIVERGED")
            break


        # Save trajectory data

        if t >= traj_data.shape[0]:
            traj_data = np.append(traj_data, np.zeros((50000, 9)),0)

        traj_data[t] = np.array([t, theta, sigma, J, gradK, gradSigma, gradMixed, alpha, updateGradSigma])


        # Display
        if verbose:
            if t % 100 == 0:
                print("\nT\t\tTheta\t\tSigma\t\tJ\t\t\tgradK\t\tgradSigma\t\tgradMixed\t\talpha\tupdateGradSigma\tw\n")
            print("{}\t\t{:.5f}\t\t{:.4f}\t\t{:.3E}\t\t{:.3f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4E}\t{:.4f}".format(t, theta, sigma, J, gradK, gradSigma, gradMixed, alpha, updateGradSigma))

        # Update parameters

        theta += alpha * gradK

        # How to update the sigma
        if two_steps:
            sigma = sigma_fun.eval()

            sigma_vect = np.linspace(MIN_SIGMA, MAX_SIGMA, 1000)
            grads = [get_gradient_sigma(theta, s, lambda_coeff, lqg_env.gamma, ENV_R, ENV_Q, lqg_env.max_pos)[0] for s in sigma_vect]

            new_sigma = sigma_vect[np.abs(grads).argmin()]

            sigma_fun.update(1, new_sigma - sigma)
        else:
            sigma_fun.update(alphaSigma, updateGradSigma)


    if filename is not None:
        np.save(filename, traj_data[:t])
        np.save(filename[:-4] + '_params.npy', initial_configuration)


def run_multiple_experiments(params, base_folder, num_iterations=10000):
    maybe_make_dir(base_folder)

    for param in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), param))

        if 'two_steps' in param_dict:
            print("Sigma: {sigma_fun}, Theta: {theta}, alphaSigma: {alphaSigma}, lambda_coeff: {lambda_coeff} two steps".format(**param_dict))
            filename = os.path.join(base_folder, "exp_{sigma_fun}_{theta}_{alphaSigma}_{lambda_coeff}_two_steps.npy".format(**param_dict))
        else:
            print("Sigma: {sigma_fun}, Theta: {theta}, alphaSigma: {alphaSigma}, lambda_coeff: {lambda_coeff}".format(**param_dict))
            filename = os.path.join(base_folder, "exp_{sigma_fun}_{theta}_{alphaSigma}_{lambda_coeff}.npy".format(**param_dict))

        run_experiment(**param_dict, filename=filename, verbose=False, n_iterations=num_iterations)

if __name__ == '__main__':
    l = Identity(1)
    run_experiment(sigma_fun=l, theta=-0.1, alphaSigma=0.00002, lambda_coeff=0.00005, n_iterations=100000, verbose=False, two_steps=False)
    exit()

    params = {
        #'sigma_fun' : [Exponential(0), Identity(1)],
        'sigma_fun' : [Identity(1), Exponential(0)],
        'theta' : [-0.1],
        'alphaSigma' : [0, 0.005, 0.01, 0.001, 0.0005, 0.02],
        #'alphaSigma' : [1],
        'lambda_coeff' : [0, 0.000001, 0.00001, 0.000005, 1]
        #,'two_steps' : [True]
    }

    base_folder = 'experiments_sigma_lambda2'

    run_multiple_experiments(params, base_folder, 100000)
