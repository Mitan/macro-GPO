import numpy as np

from src.DatasetUtils import GetAllMeasurements, GetMaxValues, GetAccumulatedRewards
from src.enum.MetricsEnum import MetricsEnum
from src.enum.SinglePointMethodsDict import single_point_methods
from src.metric.DatasetScaleExtractor import DatasetScaleExtractor


class ResultCalculator:

    def __init__(self, dataset_type, seeds, root_path, time_slot):
        self.dataset_type = dataset_type
        self.time_slot = time_slot
        self.root_path = root_path
        self.seeds = seeds

    @staticmethod
    def __get_results_for_one_seed(measurements, metric_type, batch_size, model_scale):

        if metric_type == MetricsEnum.SimpleRegret:
            max_found_values = GetMaxValues(measurements, batch_size)
            results = model_scale - max_found_values
        elif metric_type == MetricsEnum.TotalReward:
            accumulated_reward = GetAccumulatedRewards(measurements, batch_size)
            steps = 20 / batch_size
            scaled_model_mean = np.array([(1 + batch_size * i) * model_scale for i in range(steps + 1)])
            results_normaliser = np.array([1 + batch_size * i for i in range(20 / batch_size + 1)])
            results = np.divide(accumulated_reward - scaled_model_mean, results_normaliser)
        else:
            raise Exception("Unknown plotting type")
        return results

    def _get_results_for_one_method(self, method, batch_size, model_scale, metric_type):

        steps = 20 / batch_size

        len_seeds = len(self.seeds)
        all_results = np.zeros((len_seeds, steps + 1))

        for ind, seed in enumerate(self.seeds):
            seed_folder = self.root_path + 'seed' + str(seed) + '/'
            measurements = GetAllMeasurements(seed_folder, method, batch_size)
            model_seed_scale = model_scale if isinstance(model_scale, (int, long, float)) \
                else model_scale[seed]
            all_results[ind, :] = self.__get_results_for_one_seed(measurements=measurements,
                                                                  metric_type=metric_type,
                                                                  batch_size=batch_size,
                                                                  model_scale=model_seed_scale)

        error_bars = np.std(all_results, axis=0) / np.sqrt(len_seeds)
        means = np.mean(all_results, axis=0)

        return means.tolist(), error_bars.tolist()

    def calculate_results(self, batch_size, methods, method_names, metric_type):

        scale_extractor = DatasetScaleExtractor(dataset_type=self.dataset_type,
                                                time_slot=self.time_slot,
                                                batch_size=batch_size)
        # can be dict or float
        model_scale = scale_extractor.extract_mean_or_max(root_folder=self.root_path,
                                                          seeds=self.seeds,
                                                          metric_type=metric_type)

        results = []

        for index, method in enumerate(methods):
            # todo hack
            adjusted_batch_size = 1 if method in single_point_methods else batch_size

            means, error_bars = self._get_results_for_one_method(method=method,
                                                                 batch_size=adjusted_batch_size,
                                                                 model_scale=model_scale,
                                                                 metric_type=metric_type)
            results.append([method_names[index], means, error_bars])

        return results
