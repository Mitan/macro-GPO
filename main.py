from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.enum.MethodEnum import Methods
from src.run.MethodDescriptor import MethodDescriptor
from src.run.Run import run

if __name__ == '__main__':

    # batch size
    batch_size = 4

    # total budget of function evaluations
    total_budget = 20

    # seeds for evaluation
    seeds = range(3)

    # threshold for selecting the number of random macro-actions. Selected when
    ma_threshold = 20

    # type of dataset - currently Simulated, Road and Robot are supported
    dataset_type = DatasetEnum.Robot

    # load or generate dataset. If "Load" is selected, the root dataset folder should be specified
    dataset_mode = DatasetModeEnum.Load

    # the folder containing the dataset
    # if None, the current seed folder will be used (e.g. for loading the simulated dataset)
    dataset_root_folder = './new_datasets/robot/'

    results_save_root_folder = './new_tests/robot/'

    # list of methods to run
    methods = [MethodDescriptor(method_type=Methods.Exact,
                                h=2,
                                beta=0.0,
                                num_samples=2)]
    run(batch_size=batch_size,
        total_budget=total_budget,
        seeds=seeds,
        dataset_type=dataset_type,
        dataset_mode=dataset_mode,
        dataset_root_folder=dataset_root_folder,
        results_save_root_folder=results_save_root_folder,
        ma_threshold=ma_threshold,
        methods=methods)
