import multiprocessing
from exp_lqg_budget import run, AVAILABLE_EXPERIMENTS
import utils
import math

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


if __name__ == '__main__':
    run_all()
