import os

from src.TreePlanTester import testWithFixedParameters
from src.dataset_model.DatasetGenerator import DatasetGenerator
from src.enum.MetricsEnum import MetricsEnum
from src.metric.ResultCalculator import ResultCalculator


class MethodRunner:

    def __init__(self, dataset_type, dataset_root_folder, dataset_mode, batch_size):
        self.dataset_root_folder = dataset_root_folder
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.dataset_mode = dataset_mode

        self.dataset_generator = DatasetGenerator(dataset_type=self.dataset_type,
                                                  dataset_mode=self.dataset_mode,
                                                  batch_size=self.batch_size,
                                                  dataset_root_folder=self.dataset_root_folder)

    def run(self,
            total_budget,
            seeds,
            results_save_root_folder,
            ma_threshold,
            methods):
        try:
            os.makedirs(results_save_root_folder)
        except OSError:
            if not os.path.isdir(results_save_root_folder):
                raise

        for seed in seeds:
            seed_save_folder = results_save_root_folder + "seed" + str(seed) + "/"

            self._run_single_seed_test(seed_save_folder=seed_save_folder,
                                       seed=seed,
                                       total_budget=total_budget,
                                       ma_treshold=ma_threshold,
                                       methods=methods)

    def _run_single_seed_test(self,
                              seed_save_folder,
                              seed,
                              total_budget,
                              ma_treshold,
                              methods):
        try:
            os.makedirs(seed_save_folder)
        except OSError:
            if not os.path.isdir(seed_save_folder):
                raise

        m = self.dataset_generator.get_dataset_model(seed_folder=seed_save_folder,
                                                     seed=seed,
                                                     ma_treshold=ma_treshold)

        filename_rewards = seed_save_folder + "reward_histories.txt"

        if os.path.exists(filename_rewards):
            append_write = 'a'
        else:
            append_write = 'w'

        output_rewards = open(filename_rewards, append_write)

        for method in methods:
            current_res = testWithFixedParameters(model=m,
                                                  method=method.method_type,
                                                  horizon=method.h,
                                                  total_budget=total_budget,
                                                  save_folder="{}{}/".format(seed_save_folder,
                                                                             method.method_folder_name),
                                                  num_samples=method.num_samples,
                                                  beta=method.beta)

            output_rewards.write(method.method_folder_name + '\n')
            output_rewards.write(str(current_res) + '\n')
        output_rewards.close()

    def calculate_results(self,
                          methods,
                          seeds,
                          total_budget,
                          results_save_root_folder,
                          metrics=(MetricsEnum.AverageTotalReward, MetricsEnum.SimpleRegret)):
        result_calculator = ResultCalculator(dataset_type=self.dataset_type,
                                             results_save_root_folder=results_save_root_folder,
                                             seeds=seeds,

                                             total_budget=total_budget)
        results = []

        for metric in metrics:
            metric_results = result_calculator.calculate_results(batch_size=self.batch_size,
                                                                 methods=methods,
                                                                 metric_type=metric,
                                                                 dataset_root_folder=self.dataset_root_folder)
            results.append([metric, metric_results])

        self._write_results_to_file(filename="{}results.txt".format(results_save_root_folder),
                                    results=results)

    @staticmethod
    def _write_results_to_file(filename, results):
        if os.path.exists(filename):
            append_write = 'a'
        else:
            append_write = 'w'

        with open(filename, append_write) as f:
            for result in results:
                metric_string = "simple regret: " if result[0] == 2 else "average reward: "
                method_name = result[1][0][0]
                means = result[1][0][1]
                error_bars = result[1][0][2]
                f.write("{}\n".format(method_name))
                f.write("\t{} means and error bars\n".format(metric_string))
                f.write("\t\t{}\n".format(" ".join(map(str, list(means)))))
                f.write("\t\t{}\n".format(" ".join(map(str, list(error_bars)))))
