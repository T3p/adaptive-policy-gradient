from enum import Enum
import lqg1d
import numpy as np
import math
import itertools
import os

class Constants(Enum):
    ONLY_THETA = 0
    THETA_AND_SIGMA = 1
    THETA_AND_MIXED = 2

    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name

MIN_SIGMA = 0.1


def my_range(r):
    n = 0
    while n != r:
        yield n
        n += 1

def computeLoss(R, M, gamma, volume, sigma):
    return float(R*M**2)/((1-gamma)**2*sigma**2)* \
        (float(volume)/math.sqrt(2*math.pi*sigma**2) + \
            float(gamma)/(2*(1-gamma)))


def run_experiment( experiment_type=Constants.ONLY_THETA,
                    theta=-0.1,
                    sigma=20,
                    alphaSigma=0.0001,
                    n_iterations=-1,
                    eps=1e-03,
                    filename=None,
                    verbose=True):

    initial_configuration = np.array([experiment_type.value, theta, sigma, alphaSigma, n_iterations, eps, filename])
    lqg_env = lqg1d.LQG1D()

    # Hyperparameters for LQG
    R = np.asscalar(lqg_env.Q*lqg_env.max_pos**2 + lqg_env.R*lqg_env.max_action**2)
    M = lqg_env.max_pos
    gamma = lqg_env.gamma
    volume = 2*lqg_env.max_action

    traj_data = np.zeros((1, 8))

    for t in my_range(n_iterations):
        J = lqg_env.computeJ(theta, sigma)

        gradK = lqg_env.grad_K(theta, sigma)
        if abs(gradK) <= eps:
            break

        gradSigma = lqg_env.grad_Sigma(theta, sigma)
        gradMixed = lqg_env.grad_mixed(theta, sigma)

        c = computeLoss(R, M, gamma, volume, sigma)
        alpha=1/(2*c)

        if verbose:
            if t % 100 == 0:
                print("\nT\t\tTheta\t\tSigma\t\tJ\t\t\tgradK\t\tgradSigma\t\tgradMixed\t\talpha\n")
            print("{}\t\t{:.5f}\t\t{:.4f}\t\t{:.3E}\t\t{:.3f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4E}".format(t, theta, sigma, J, gradK, gradSigma, gradMixed, alpha))

        if t == 0:
            traj_data = np.array([t, theta, sigma, J, gradK, gradSigma, gradMixed, alpha])
        else:
            traj_data = np.vstack([traj_data, np.array([t, theta, sigma, J, gradK, gradSigma, gradMixed, alpha])])

        theta += alpha * gradK

        if experiment_type == Constants.THETA_AND_SIGMA:
            sigma += alphaSigma * gradSigma
        elif experiment_type == Constants.THETA_AND_MIXED:
            sigma += alphaSigma * gradMixed

        # Bound the sigma to be always positive
        sigma = max(MIN_SIGMA, sigma)

    if filename is not None:
        np.save(filename, traj_data)
        np.save(filename[:-4] + '_params.npy', initial_configuration)


    #print("Best theta: {}".format(lqg_env.computeOptimalK()))
    #print("Difference from optimal theta: {}".format(abs(lqg_env.computeOptimalK() - theta)))

if __name__ == '__main__':
    params = {
        'sigma' : [1, 10, 20, 30],
        'theta' : [-0.1, -0.2, -0.01],
        'alphaSigma' : [0.1, 0.01, 0.001, 0.0001],
        'experiment_type' : [Constants.ONLY_THETA, Constants.THETA_AND_SIGMA, Constants.THETA_AND_MIXED]
    }

    base_folder = 'experiments_sigma'

    for param in itertools.product(*params.values()):
        param_dict = dict(zip(params.keys(), param))
        print("Sigma: {sigma}, Theta: {theta}, alphaSigma: {alphaSigma}, experiment_type: {experiment_type}".format(**param_dict))

        filename = os.path.join(base_folder, "exp_{sigma}_{theta}_{alphaSigma}_{experiment_type}.npy".format(**param_dict))

        run_experiment(**param_dict, filename=filename, verbose=False, n_iterations=10000)
