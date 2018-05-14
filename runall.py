import multiprocessing
# from exp_lqg_budget import run, AVAILABLE_EXPERIMENTS
from exp_adaptive_exploration import run
import utils
import math
import numpy as np
import itertools

def run_all():
    BATCH_SIZE = 300
    MAX_ITERS = 10000
    FILEPATH = 'EXPERIMENTS_FINAL_LONG'

    utils.maybe_make_dir(FILEPATH)

    args = []
    #for exp_name, _ in AVAILABLE_EXPERIMENTS.items():
    for exp_name in ['MonotonicOnlyTheta','MonotonicThetaAndSigma', 'MonotonicZeroBudgetEveryStep', 'NoWorseThanBaselineEveryStep', 'ExpBudget_NoDetPolicy', 'ExpBudget_SemiDetPolicy', 'ExpBudget_DetPolicy']:
        for random_seed in [1, 2, 3, 4, 5]:
            name = exp_name + '_' + str(random_seed)

            args.append([exp_name, name, BATCH_SIZE, MAX_ITERS, FILEPATH, random_seed, False, True])

    for exp_name in ['SimultaneousThetaAndSigma_half', 'SimultaneousThetaAndSigma_two_thirds_theta', 'SimultaneousThetaAndSigma_two_thirds_sigma']:
        for random_seed in [1, 2, 3, 4, 5]:
            name = exp_name + '_' + str(random_seed)

            args.append([exp_name, name, math.floor(BATCH_SIZE*2/3), MAX_ITERS, FILEPATH, random_seed, False, True])


    p = multiprocessing.Pool()
    p.starmap(run, args)

def run_grid():
    BATCH_SIZE = 100
    MAX_ITERS = 10000
    FILEPATH = 'GRID_LQG'

    utils.maybe_make_dir(FILEPATH)

    THETA_INIT = np.array([-0.1])
    ALGORITHMS = ['MonotonicOnlyTheta', 'ExpBudget_NoDetPolicy', 'NoWorseThanBaselineEveryStep']
    INITIAL_SIGMA = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    ALPHA = [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    RANDOM_SEED = [1, 2, 3, 4, 5]

    args = []

    for exp, sigma, seed in itertools.product(ALGORITHMS, INITIAL_SIGMA, RANDOM_SEED):
        name = exp + '_' + str(sigma) + '_' + str(seed)

        args.append([exp, 'LQG1D-v0', name, BATCH_SIZE, MAX_ITERS, FILEPATH, seed, False, True, 1, sigma, THETA_INIT, None])

    for sigma, alpha, seed in itertools.product(INITIAL_SIGMA, ALPHA, RANDOM_SEED):
        exp = 'MonotonicNaiveGradient'
        name = exp + '_' + str(alpha) + '_' + str(sigma) + '_' + str(seed)

        args.append([exp, 'LQG1D-v0', name, BATCH_SIZE, MAX_ITERS, FILEPATH, seed, False, True, 1, sigma, THETA_INIT, alpha])

    for sigma, alpha, seed in itertools.product(INITIAL_SIGMA, ALPHA, RANDOM_SEED):
        exp = 'Adam'
        name = exp + '_' + str(alpha) + '_' + str(sigma) + '_' + str(seed)

        args.append([exp, 'LQG1D-v0', name, BATCH_SIZE, MAX_ITERS, FILEPATH, seed, False, True, 1, sigma, THETA_INIT, alpha])




    p = multiprocessing.Pool()
    p.starmap(run, args)


if __name__ == '__main__':
    # run_all()
    run_grid()
