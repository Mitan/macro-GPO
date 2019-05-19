import sys

from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.enum.MethodEnum import Methods
from src.run import run
from src.scripts.betaTest import hack_script_beta

# from src.scripts.main_testing_script import hack_script

if __name__ == '__main__':
    start = int(sys.argv[1])
    h = int(sys.argv[2])
    # start = 100
    hack_script_beta(start, h)

    # list of planning horizon values
    h_list = [2]

    # batch size
    batch_size = 4

    # total budget of function evaluations
    total_budget = 20

    # number of stochastic samples generated for every node
    num_samples = 500

    # exploration parameter beta
    beta = 0.0

    # seeds for evaluation
    seeds = range(10)

    # threshold for selecting the number of random macro-actions. Selected when
    ma_threshold = 20

    # type of dataset - currently Simulated, Road and Robot are supported
    dataset_type = DatasetEnum.Simulated

    # load or generate dataset. If "Load" is selected, the root dataset folder should be specified
    dataset_mode = DatasetModeEnum.Load

    # execute exact or anytime algorithm
    method = Methods.Exact

    # the root dataset folder should be specified for loading the dataset
    dataset_root_folder = './'

    results_save_root_folder = './'

    run(h_list=h_list,
        batch_size=batch_size,
        total_budget=total_budget,
        num_samples=num_samples,
        beta=beta,
        seeds=seeds,
        dataset_type=dataset_type,
        dataset_mode=dataset_mode,
        dataset_root_folder=dataset_root_folder,
        results_save_root_folder=results_save_root_folder,
        ma_threshold=ma_threshold,
        method=method)


