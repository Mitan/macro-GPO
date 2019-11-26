# from src.Utils import branin_transform
from src.enum.DatasetEnum import DatasetEnum
from src.enum.MetricsEnum import MetricsEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.dataset_model.DatasetGenerator import DatasetGenerator


class DatasetScaleExtractor:

    def __init__(self, dataset_type, time_slot, batch_size):
        self.dataset_generator = DatasetGenerator(dataset_type,
                                                  dataset_mode=DatasetModeEnum.Load,
                                                  time_slot=time_slot, batch_size=batch_size)
        self.type = dataset_type

    def _extract_mean(self, root_folder, seeds):
        empirical_mean = self.dataset_generator.hyper_storer.empirical_mean
        if not empirical_mean:
            return self.__extract_non_constant_mean(root_folder, seeds)

        # if self.type == DatasetEnum.Branin:
        #
        #     empirical_mean =  branin_transform(empirical_mean)
        # print  empirical_mean
        return empirical_mean

    def extract_mean_or_max(self, root_folder, seeds, metric_type):
        if metric_type == MetricsEnum.SimpleRegret:
            return self._extract_max(root_folder, seeds)
        elif metric_type == MetricsEnum.AverageTotalReward:
            return self._extract_mean(root_folder, seeds)
        else:
            raise Exception("Unknown metric type")

    def _extract_max(self, root_folder, seeds):
        max_value = self.dataset_generator.hyper_storer.max_value
        if not max_value:
            return self.__extract_non_constant_max(root_folder, seeds)

        # if self.type == DatasetEnum.Branin:
        #     max_value = branin_transform(max_value)

        return max_value

    # a case when there are multiple datasets (e.g. simulated) so each realisation has a different max
    def __extract_non_constant_max(self, root_folder, seeds):
        model_max_values = {}
        # reachable_locations = generate_set_of_reachable_locations(b_size=batch_size, start=(1.0, 1.0), gap=0.05)
        for seed in seeds:
            seed_dataset_path = root_folder + 'seed' + str(seed) + '/'
            m = self.dataset_generator.get_dataset_model(root_folder=seed_dataset_path, seed=seed, ma_treshold=None)
            # reachable_max = max(map(lambda x: m(x), reachable_locations))
            global_max = m.get_max()

            model_max_values[seed] = global_max
        return model_max_values

    # a case when there are multiple datasets (e.g. simulated) so each realisation has a different mean
    def __extract_non_constant_mean(self, root_folder, seeds):
        model_mean_values = {}

        for seed in seeds:
            seed_dataset_path = root_folder + 'seed' + str(seed) + '/'
            m = self.dataset_generator.get_dataset_model(root_folder=seed_dataset_path, seed=seed, ma_treshold=None)

            model_mean_values[seed] = m.get_empirical_mean()
        return model_mean_values
