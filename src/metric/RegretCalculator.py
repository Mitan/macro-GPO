import numpy as np
from src.DatasetUtils import GetAllMeasurements, GetMaxValues
from src.enum.SinglePointMethodsDict import single_point_methods
from src.metric.DatasetMaxExtractor import DatasetMaxExtractor
from src.plotting.ResultsPlotter import PlotData


class RegretCalculator:

    def __init__(self, dataset_type,seeds, root_path, time_slot):
        self.dataset_type = dataset_type
        self.time_slot = time_slot
        self.root_path = root_path
        self.seeds = seeds

    def __calculate_regrets_for_one_method(self, method, batch_size, model_max):

        steps = 20 / batch_size

        # +1 for initial stage
        results_for_method = np.zeros((steps + 1,))

        len_seeds = len(self.seeds)
        all_regrets = np.zeros((len_seeds, steps + 1))

        for ind, seed in enumerate(self.seeds):
            seed_folder = self.root_path + 'seed' + str(seed) + '/'
            measurements = GetAllMeasurements(seed_folder, method, batch_size)
            max_found_values = GetMaxValues(measurements, batch_size)
            assert max_found_values.shape == results_for_method.shape
            # hack to distinghiush the cases where the model_max is a dictionary (for simulated realisations)
            #  or a float (when there's a one dataset)
            max_of_seed_dataset = model_max if isinstance(model_max, (int, long, float)) \
                else model_max[seed]
            all_regrets[ind, :] = max_of_seed_dataset - max_found_values

        error_bars = np.std(all_regrets, axis=0) / np.sqrt(len_seeds)

        means = np.mean(all_regrets, axis=0)
        return means.tolist(), error_bars.tolist()

    def process_regrets(self, batch_size, methods, method_names,
                        output_filename, plottingType, plot_bars):

        max_extractor = DatasetMaxExtractor(dataset_type=self.dataset_type,
                                            time_slot=self.time_slot,
                                            batch_size=batch_size)
        model_max = max_extractor.extract_max(root_folder=self.root_path, seeds=self.seeds)

        results = []

        # regret_calculator = RegretCalculator(seeds=seeds, root_path=root_path, model_max=model_max)

        for index, method in enumerate(methods):
            # todo hack
            adjusted_batch_size = 1 if method in single_point_methods else batch_size

            regrets, error_bars = self.__calculate_regrets_for_one_method(method=method,
                                                                          batch_size=adjusted_batch_size,
                                                                          model_max=model_max)
            results.append([method_names[index], regrets, error_bars])

        PlotData(results=results, output_file_name=output_filename,
                 plottingType=plottingType, dataset=self.dataset_type, plot_bars=plot_bars)
        return results
