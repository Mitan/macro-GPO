from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.model.DomainDescriptorFactory import get_domain_descriptor
from src.model.HypersStorer import RobotHypersStorer_2, RobotHypersStorer_16, RoadHypersStorer_Log18, \
    RoadHypersStorer_Log44
from src.model.RoadMapValueDict import RoadMapValueDict
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
        elif self.type == DatasetEnum.Road:
            return self.__get_road_dataset_model()
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

        domain_descriptor = get_domain_descriptor(DatasetEnum.Robot)

        m = RobotValueDict(data_filename=data_filename, coords_filename=coords_filename,
                           neighbours_filename=neighbours_filename, hyper_storer=hyper_storer,
                           domain_descriptor=domain_descriptor)
        return m

    def __get_road_dataset_model(self):
        if self.mode == DatasetModeEnum.Generate:
            raise ValueError("Generate mode is available only for simulated dataset")

        filename = '../../datasets/slot' + str(self.time_slot) + '/tlog' + str(self.time_slot) + '.dom'

        if self.time_slot == 44:
            hyper_storer = RoadHypersStorer_Log44()
        elif self.time_slot == 18:
            hyper_storer = RoadHypersStorer_Log18()
        else:
            raise Exception("wrong taxi time slot")

        domain_descriptor = get_domain_descriptor(DatasetEnum.Road)
        m = RoadMapValueDict(filename=filename, hyper_storer=hyper_storer, domain_descriptor=domain_descriptor)
        return m
