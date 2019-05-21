from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.enum.MethodEnum import Methods
from src.run.MethodDescriptor import MethodDescriptor
from src.run.MethodRunner import MethodRunner

if __name__ == '__main__':
    # batch size
    batch_size = 4

    # type of dataset - currently Simulated, Road and Robot are supported
    dataset_type = DatasetEnum.Simulated

    # load or generate dataset. If "Load" is selected, the root dataset folder should be specified
    dataset_mode = DatasetModeEnum.Load

    # the folder containing the dataset
    # if None, the current seed folder will be used (e.g. for loading the simulated dataset)
    dataset_root_folder = './new_datasets/robot/'
    dataset_root_folder = None

    method_runner = MethodRunner(dataset_type=dataset_type,
                                 dataset_mode=dataset_mode,
                                 batch_size=batch_size,
                                 dataset_root_folder=dataset_root_folder)

    results_save_root_folder = './new_tests/simulated/'

    # total budget of function evaluations
    total_budget = 20

    # seeds for evaluation
    seeds = range(3)

    # threshold for selecting the number of random macro-actions. Selected when
    ma_threshold = 20

    # list of methods to run
    methods = [MethodDescriptor(method_type=Methods.Exact,
                                h=2,
                                beta=20.0,
                                num_samples=2),
               MethodDescriptor(method_type=Methods.Exact,
                                h=1,
                                beta=1.0,
                                num_samples=2)
               ]

    method_runner.run(total_budget=total_budget,
                      seeds=seeds,
                      results_save_root_folder=results_save_root_folder,
                      ma_threshold=ma_threshold,
                      methods=methods)
