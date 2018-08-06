import numpy as np
from src.DatasetUtils import GetAllMeasurements, GetMaxValues


class RegretCalculator:

    def __init__(self, seeds, root_path, model_max):
        self.model_max = model_max
        self.root_path = root_path
        self.seeds = seeds

    def calculate_regrets_for_one_method(self, method, batch_size):

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
            max_of_seed_dataset = self.model_max if isinstance(self.model_max, (int, long, float)) \
                else self.model_max[seed]
            all_regrets[ind, :] = max_of_seed_dataset - max_found_values

        error_bars = np.std(all_regrets, axis=0) / np.sqrt(len_seeds)

        means = np.mean(all_regrets, axis=0)
        return means.tolist(), error_bars.tolist()