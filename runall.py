import multiprocessing
from exp_lqg_budget import run, AVAILABLE_EXPERIMENTS

def run_all():
    BATCH_SIZE = 900
    MAX_ITERS = 10000
    FILEPATH = 'EXPERIMENTS_FINAL_1'

    args = []
    for exp_name, _ in AVAILABLE_EXPERIMENTS.items():
        for random_seed in [1, 2, 3, 4, 5]:
            name = exp_name + '_' + str(random_seed)

            args.append([exp_name, name, BATCH_SIZE, MAX_ITERS, FILEPATH, random_seed, False, False])


    p = multiprocessing.Pool()
    p.starmap(run, args)


if __name__ == '__main__':
    run_all()
