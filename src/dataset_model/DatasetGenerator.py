from src.dataset_model.BraninMapValueDict import BraninMapValueDict
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum
from src.dataset_model.DomainDescriptorFactory import get_domain_descriptor
from src.dataset_model.HypersStorerFactory import RobotHypersStorer_2, RobotHypersStorer_16, RoadHypersStorer_Log18, \
    RoadHypersStorer_Log44, get_hyper_storer
from src.dataset_model.RoadMapValueDict import RoadMapValueDict
from src.dataset_model.RobotMapValueDict import RobotValueDict
from src.dataset_model.SimulatedMapValueDict import SimulatedMapValueDict


class DatasetGenerator:

    def __init__(self, dataset_type, dataset_mode, time_slot, batch_size):
        self.type = dataset_type
        self.mode = dataset_mode
        self.time_slot = time_slot
        self.batch_size = batch_size
        self.hyper_storer = get_hyper_storer(self.type, self.time_slot)

    def get_dataset_model(self, root_folder, seed, ma_treshold):
        # select_all select all macro-actions
        if self.type == DatasetEnum.Robot:
            return self.__get_robot_dataset_model(root_folder, ma_treshold)
        elif self.type == DatasetEnum.Road:
            return self.__get_road_dataset_model(root_folder, ma_treshold)
        elif self.type == DatasetEnum.Simulated:
            return self.__get_simulated_dataset_model(root_folder, seed)
        elif self.type == DatasetEnum.Branin:
            return self.__get_branin_dataset_model(root_folder)
        else:
            raise ValueError("Unknown dataset")
            # private methods

    def __get_robot_dataset_model(self, root_folder, ma_treshold):

        data_filename = '../../datasets/robot/selected_slots/slot_' + str(self.time_slot) + '/noise_final_slot_' + \
                        str(self.time_slot) + '.txt'
        neighbours_filename = '../../datasets/robot/all_neighbours.txt'
        coords_filename = '../../datasets/robot/all_coords.txt'

        # hyper_storer = get_hyper_storer(DatasetEnum.Robot, self.time_slot)

        domain_descriptor = get_domain_descriptor(DatasetEnum.Robot)

        m = RobotValueDict(data_filename=data_filename, coords_filename=coords_filename,
                           neighbours_filename=neighbours_filename, hyper_storer=self.hyper_storer,
                           domain_descriptor=domain_descriptor, batch_size=self.batch_size)

        location_filename = root_folder + 'start_location.txt'

        actions_filename = root_folder + 'actions_selected.txt'

        if self.mode == DatasetModeEnum.Generate:
            m.GenerateStartLocation()
            m.SelectMacroActions(actions_filename=actions_filename, ma_treshold=ma_treshold)
            with open(location_filename, 'w') as f:
                f.write(str(m.start_location[0, 0]) + " " + str(m.start_location[0, 1]))
            print "Generating start location and macro-actions"
        else:
            m.LoadStartLocation(location_filename)
            m.LoadSelectedMacroactions(actions_filename=actions_filename)
            print "Loading start location and macro-actions"
        return m

    def __get_road_dataset_model(self, root_folder, ma_treshold):

        filename = '../../datasets/slot' + str(self.time_slot) + '/tlog' + str(self.time_slot) + '.dom'

        # hyper_storer = get_hyper_storer(DatasetEnum.Road, self.time_slot)

        domain_descriptor = get_domain_descriptor(DatasetEnum.Road)
        m = RoadMapValueDict(filename=filename,
                             hyper_storer=self.hyper_storer,
                             domain_descriptor=domain_descriptor,
                             batch_size=self.batch_size)

        location_filename = root_folder + 'start_location.txt'

        actions_filename = root_folder + 'actions_selected.txt'

        if self.mode == DatasetModeEnum.Generate:
            m.GenerateStartLocation()
            m.SelectMacroActions(actions_filename=actions_filename, ma_treshold=ma_treshold)
            with open(location_filename, 'w') as f:
                f.write(str(m.start_location[0, 0]) + " " + str(m.start_location[0, 1]))
            print "Generating start location and macro-actions"
        else:
            m.LoadStartLocation(location_filename)
            m.LoadSelectedMacroactions(actions_filename=actions_filename)
            print "Loading start location and macro-actions"
        return m

    def __get_branin_dataset_model(self, root_folder):
        # hyper_storer = get_hyper_storer(DatasetEnum.Simulated, self.time_slot)

        domain_descriptor = get_domain_descriptor(DatasetEnum.Branin)

        location_filename = root_folder + 'start_location.txt'

        print "Loading model"
        dataset_filename = './datasets/branin/branin_1600points_inverse_sign_normalised_ok.txt'

        dataset_filename = './datasets/branin/branin_400points_inverse_sign_normalised.txt'

        dataset_filename = './datasets/branin/camel_600points_inverse_sign_normalised.txt'

        m = BraninMapValueDict(hyper_storer=self.hyper_storer,
                               domain_descriptor=domain_descriptor,
                               filename=dataset_filename,
                               batch_size=self.batch_size)

        if self.mode == DatasetModeEnum.Generate:
            m.GenerateStartLocation()
            with open(location_filename, 'w') as f:
                f.write(str(m.start_location[0, 0]) + " " + str(m.start_location[0, 1]))
        else:
            m.LoadStartLocation(location_filename)

        return m

    def __get_simulated_dataset_model(self, root_folder, seed):
        # hyper_storer = get_hyper_storer(DatasetEnum.Simulated, self.time_slot)

        domain_descriptor = get_domain_descriptor(DatasetEnum.Simulated)

        location_filename = root_folder + 'start_location.txt'

        if self.mode == DatasetModeEnum.Generate:
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor,
                                      seed=seed,
                                      batch_size=self.batch_size)
            m.WriteToFile(root_folder + "dataset.txt")
            m.GenerateStartLocation()

            with open(location_filename, 'w') as f:
                f.write(str(m.start_location[0, 0]) + " " + str(m.start_location[0, 1]))
        else:
            # print "Loading model"
            dataset_filename = root_folder + 'dataset.txt'
            # dataset_filename = '../../datasets/branin/branin_1600points_inverse_sign_normalised.txt'
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor,
                                      filename=dataset_filename,
                                      batch_size=self.batch_size)

            m.LoadStartLocation(location_filename)
        return m
