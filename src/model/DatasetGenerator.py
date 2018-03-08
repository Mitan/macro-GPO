from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.model.HypersStorer import RobotHypersStorer_2, RobotHypersStorer_16
from src.model.RobotMapValueDict import RobotValueDict


class DatasetGenerator:

    def __init__(self, dataset_type, dataset_mode, time_slot):
        self.type = dataset_type
        self.mode = dataset_mode
        self.time_slot = time_slot

    def get_dataset_model(self):
        # select_all select all macro-actions
        if self.type == DatasetEnum.Robot:
            return self.__get_robot_dataset_model()
        else:
            raise ValueError("Unknown dataset")
            # private methods

    def __get_robot_dataset_model(self):
        if self.mode == DatasetModeEnum.Generate:
            raise ValueError("Generate mode is available only for simulated dataset")

        data_filename = '../../datasets/robot/selected_slots/slot_' + str(self.time_slot) + '/noise_final_slot_' + \
                        str(self.time_slot) + '.txt'
        neighbours_filename = '../../datasets/robot/all_neighbours.txt'
        coords_filename = '../../datasets/robot/all_coords.txt'

        if self.time_slot == 2:
            hyper_storer = RobotHypersStorer_2()
        elif self.time_slot == 16:
            hyper_storer = RobotHypersStorer_16()
        else:
            raise Exception("wrong robot time slot")

        m = RobotValueDict(data_filename, coords_filename, neighbours_filename, hyper_storer)
        return m
