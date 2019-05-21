from config import Config
from src.run.MethodRunner import MethodRunner

if __name__ == '__main__':

    method_runner = MethodRunner(dataset_type=Config.DATASET_TYPE,
                                 dataset_mode=Config.DATASET_MODE,
                                 batch_size=Config.BATCH_SIZE,
                                 dataset_root_folder=Config.DATASET_ROOT_FOLDER)
    """
    method_runner.run(total_budget=Config.TOTAL_BUDGET,
                      seeds=Config.SEEDS,
                      results_save_root_folder=Config.RESULTS_SAVE_ROOT_FOLDER,
                      ma_threshold=Config.MA_THRESHOLD,
                      methods=Config.METHODS)
    """
    method_runner.calculate_results(methods=Config.METHODS,
                                    seeds=Config.SEEDS,
                                    total_budget=Config.TOTAL_BUDGET,
                                    results_save_root_folder=Config.RESULTS_SAVE_ROOT_FOLDER)
