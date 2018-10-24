from random import choice, sample

from src.dataset_model.MapValueDictBase import MapValueDictBase
import numpy as np
from src.enum.DatasetEnum import DatasetEnum

batch_road_macroactions = []


class RobotValueDict(MapValueDictBase):

    def __init__(self, data_filename, coords_filename, neighbours_filename, hyper_storer, domain_descriptor, batch_size):

        self.dataset_type = DatasetEnum.Robot
        self.hyper_storer = hyper_storer
        self.domain_descriptor = domain_descriptor
        self.batch_size = batch_size

        data_lines = np.genfromtxt(data_filename)

        self.__number_of_points = data_lines.shape[0]

        locs = data_lines[:, 1:3]

        vals = data_lines[:, 3]

        self.SetNeighbours(coords_filename=coords_filename, neighbours_filename=neighbours_filename)

        # dict of selected macroactions
        self.selected_actions_dict = None

        MapValueDictBase.__init__(self, locations=locs, values=vals)

        self.mean = np.mean(vals)

    def __IdToCoord(self, all_coords_data, id):
        id = int(id)
        assert all_coords_data[id - 1, 0] == id
        return tuple(all_coords_data[id - 1, 1:])

    def SetNeighbours(self, coords_filename, neighbours_filename):

        self.neighbours = {}

        all_coords_data = np.genfromtxt(coords_filename)
        with open(neighbours_filename, 'r') as outfile:
            all_neighbours_lines = outfile.readlines()

        for line in all_neighbours_lines:
            line = line.split()
            line = map(float, line)
            current_loc_id = line[0]

            current_loc_coord = self.__IdToCoord(all_coords_data, current_loc_id)

            id_neighbours = line[1:]
            tuple_neighbours = map(lambda x : self.__IdToCoord(all_coords_data, x), id_neighbours)
            self.neighbours[current_loc_coord] = tuple_neighbours
            # self.neighbours[current_loc_coord] = id_neighbours


    def GetNeighbours(self, location):
        tuple_loc = tuple(location)
        # return self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []
        # should not raise an exception
        return self.neighbours[tuple_loc]

    # list of 2D arrays
    def ___ExpandActions(self, start):
        # including the start, hence +1
        if len(start) == self.batch_size + 1:
            yield np.asarray(start[1:])
        else:
            current = start[-1]

            for next_node in self.GetNeighbours(current):
                # Do we need the first condition?
                if (next_node in start):
                    continue
                # print self.__call__(next_node)
                for state in self.___ExpandActions(start + [next_node]):
                    yield state

    def SelectMacroActions(self, actions_filename, ma_treshold):
        self.selected_actions_dict = {}

        actions_file  = open(actions_filename, 'w') if ma_treshold else None

        for loc in self.locations:
            all_macro_actions = self.GenerateAllMacroActions(loc)

            if not ma_treshold:
                self.selected_actions_dict[tuple(loc)] = all_macro_actions
                continue

            length = len(all_macro_actions)

            if length == 0:
                # do nothing
                continue
            elif length < ma_treshold:
                self.selected_actions_dict[tuple(loc)] = all_macro_actions
                actions_file.write(str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(range(length)) + '\n')
            else:
                generated_indexes = sample(xrange(length), ma_treshold)
                self.selected_actions_dict[tuple(loc)] = [all_macro_actions[i] for i in generated_indexes]
                actions_file.write(str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(generated_indexes) + '\n')

        if actions_file:
            actions_file.close()

    # for given state
    def GetSelectedMacroActions(self, current_state):
        current_state = tuple(current_state)
        return self.selected_actions_dict[current_state] if current_state in self.selected_actions_dict.keys() else []

    # for given state
    def GenerateAllMacroActions(self, current_state):
        current_state = tuple(current_state)
        return list(self.___ExpandActions([current_state]))

    def LoadSelectedMacroactions(self, actions_filename):

        self.selected_actions_dict = {}

        lines = open(actions_filename).readlines()

        for line in lines:
            string_numbers = line.replace(',',' ').replace('[',' ').replace(']',' ').split()
            numbers = map(float, string_numbers)
            loc = ( numbers[0], numbers[1])
            indexes = map(int, numbers[2:])
            # print loc, indexes
            all_macro_actions = self.GenerateAllMacroActions(loc)
            length = len(indexes)
            assert length <= 20

            self.selected_actions_dict[tuple(loc)] = [all_macro_actions[i] for i in indexes]

    def GenerateStartLocation(self):
        self.start_location = np.array([choice(list(self.locations))])

    def LoadStartLocation(self, location_filename):

        # should contain only one line
        string_locations = open(location_filename).readline().split()
        location = map(float, string_locations)
        assert len(location) == 2
        self.start_location = np.array([location])


if __name__ == "__main__":
    """
    filename = './taxi18.dom'
    # cannot use - cylcic linking
     m = GenerateRoadModelFromFile(filename)
    for i in m.locations:
        print i, m.GetNeighbours(i)
    
    data_file = '../datasets/robot/selected_slots/slot_2/final_slot_2.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    RobotValueDict(data_filename=data_file, coords_filename=coords_file, neighbours_filename=neighbours_file)
    """
    pass