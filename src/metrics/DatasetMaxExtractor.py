from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.model.DatasetGenerator import DatasetGenerator


class DatasetMaxExtractor:

    def __init__(self, dataset_type, time_slot, batch_size):
        self.dataset_generator = DatasetGenerator(dataset_type,
                                                  dataset_mode=DatasetModeEnum.Load,
                                                  time_slot=time_slot, batch_size=batch_size)
        self.type = dataset_type
        self.time_slot = time_slot
        self.batch_size = batch_size

    def extract_max(self, root_folder, seeds):
        # select_all select all macro-actions
        if self.type == DatasetEnum.Robot or self.type == DatasetEnum.Road:
            return self.__extract_constant_max(root_folder, seeds)
        elif self.type == DatasetEnum.Simulated:
            return self.__extract_non_constant_max()
        else:
            raise ValueError("Unknown dataset")

    # a case when there is only one dataset, so the max is constant across all the realisations
    def __extract_constant_max(self, root_folder, seeds):
        first_seed = seeds[0]
        m = self.dataset_generator.get_dataset_model(root_folder=root_folder + 'seed' + str(first_seed) + '/',
                                                     seed=first_seed,
                                                     ma_treshold=None)

        return m.GetMax()

        # a case when there are multiple datasets (e.g. simulated) so each realisation has a different max
    def __extract_non_constant_max(self):
        return 0