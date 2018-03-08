from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.model.RobotMapValueDict import RobotValueDict


class DatasetGenerator:

    def __init__(self, dataset_type, dataset_mode):
        self.type = dataset_type
        self.mode = dataset_mode

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

        time_slot = 16
        data_filename = '../datasets/robot/selected_slots/slot_' + str(time_slot) + '/noise_final_slot_' + \
                        str(time_slot) + '.txt'
        neighbours_filename = '../datasets/robot/all_neighbours.txt'
        coords_filename = '../datasets/robot/all_coords.txt'

        m = RobotValueDict(data_filename, coords_filename, neighbours_filename)
        return m
