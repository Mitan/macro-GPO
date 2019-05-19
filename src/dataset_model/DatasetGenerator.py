from src.dataset_model.DomainDescriptorFactory import get_domain_descriptor
from src.dataset_model.HypersStorerFactory import get_hyper_storer
from src.dataset_model.RoadMapValueDict import RoadMapValueDict
from src.dataset_model.RobotMapValueDict import RobotValueDict
from src.dataset_model.SimulatedMapValueDict import SimulatedMapValueDict
from src.enum.DatasetEnum import DatasetEnum
from src.enum.DatasetModeEnum import DatasetModeEnum


class DatasetGenerator:

    def __init__(self, dataset_type, dataset_mode, batch_size, dataset_root_folder):
        self.dataset_root_folder = dataset_root_folder
        self.type = dataset_type
        self.mode = dataset_mode
        self.batch_size = batch_size
        self.hyper_storer = get_hyper_storer(self.type)

    # dataset_root is the folder where datasets are stored (e.g. for Road or robot datasets)
    # seed_folder is a folder for saving (e.g. start location or selected actions)
    def get_dataset_model(self, seed_folder, seed, ma_treshold):
        # select_all select all macro-actions
        if self.type == DatasetEnum.Robot:
            return self.__get_robot_dataset_model(seed_folder=seed_folder,
                                                  ma_treshold=ma_treshold)
        elif self.type == DatasetEnum.Road:
            return self.__get_road_dataset_model(seed_folder=seed_folder,
                                                 ma_treshold=ma_treshold)
        elif self.type == DatasetEnum.Simulated:
            return self.__get_simulated_dataset_model(seed_folder=seed_folder,
                                                      seed=seed)
        else:
            raise ValueError("Unknown dataset")
            # private methods

    def __get_robot_dataset_model(self, seed_folder, ma_treshold):

        data_filename = self.dataset_root_folder + 'noise_final_slot_16.txt'
        neighbours_filename = self.dataset_root_folder + 'all_neighbours.txt'
        coords_filename = self.dataset_root_folder + 'all_coords.txt'

        domain_descriptor = get_domain_descriptor(DatasetEnum.Robot)

        m = RobotValueDict(data_filename=data_filename, coords_filename=coords_filename,
                           neighbours_filename=neighbours_filename, hyper_storer=self.hyper_storer,
                           domain_descriptor=domain_descriptor, batch_size=self.batch_size)

        location_filename = seed_folder + 'start_location.txt'

        actions_filename = seed_folder + 'actions_selected.txt'

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

    def __get_road_dataset_model(self, seed_folder, ma_treshold):

        filename = self.dataset_root_folder + 'tlog18.dom'

        # hyper_storer = get_hyper_storer(DatasetEnum.Road, self.time_slot)

        domain_descriptor = get_domain_descriptor(DatasetEnum.Road)
        m = RoadMapValueDict(filename=filename,
                             hyper_storer=self.hyper_storer,
                             domain_descriptor=domain_descriptor,
                             batch_size=self.batch_size)

        location_filename = seed_folder + 'start_location.txt'

        actions_filename = seed_folder + 'actions_selected.txt'

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

    # simulated datasets are stored in seed folders, so dataset_root_folder is None, while save folder is seed_folder
    def __get_simulated_dataset_model(self, seed_folder, seed):
        domain_descriptor = get_domain_descriptor(DatasetEnum.Simulated)

        location_filename = seed_folder + 'start_location.txt'

        if self.mode == DatasetModeEnum.Generate:
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor,
                                      seed=seed,
                                      batch_size=self.batch_size)
            m.WriteToFile(seed_folder + "dataset.txt")
            m.GenerateStartLocation()

            with open(location_filename, 'w') as f:
                f.write(str(m.start_location[0, 0]) + " " + str(m.start_location[0, 1]))
        else:
            print "Loading model"
            dataset_filename = seed_folder + 'dataset.txt'
            m = SimulatedMapValueDict(hyper_storer=self.hyper_storer,
                                      domain_descriptor=domain_descriptor,
                                      filename=dataset_filename,
                                      batch_size=self.batch_size)

            m.LoadStartLocation(location_filename)
        return m
