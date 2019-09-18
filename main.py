import sys

from config import Config
from src.run.MethodRunner import MethodRunner

if __name__ == '__main__':

    start = int(sys.argv[1])
    interval = 5
    method_runner = MethodRunner(dataset_type=Config.DATASET_TYPE,
                                 dataset_mode=Config.DATASET_MODE,
                                 batch_size=Config.BATCH_SIZE,
                                 dataset_root_folder=Config.DATASET_ROOT_FOLDER)
    method_runner.run(total_budget=Config.TOTAL_BUDGET,
                      seeds=range(start, start + interval),
                      results_save_root_folder=Config.RESULTS_SAVE_ROOT_FOLDER,
                      ma_threshold=Config.MA_THRESHOLD,
                      methods=Config.METHODS)

    # # returns result as a list. Each element is a list:
    # # metric_type, [[MethodDescriptor, means, error_bars],...]
    # results = method_runner.calculate_results(methods=Config.METHODS,
    #                                           seeds=Config.SEEDS,
    #                                           total_budget=Config.TOTAL_BUDGET,
    #                                           results_save_root_folder=Config.RESULTS_SAVE_ROOT_FOLDER,
    #                                           metrics=Config.METRICS_LIST)
    #
    # method_runner.plot_results(total_budget=Config.TOTAL_BUDGET,
    #                            results=results,
    #                            plot_bars=Config.PLOT_BARS,
    #                            results_save_root_folder=Config.RESULTS_SAVE_ROOT_FOLDER )
