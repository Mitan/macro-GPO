from StringIO import StringIO
from random import choice, sample

from GaussianProcess import MapValueDict
import numpy as np
import math

from Utils import LineToTuple

batch_road_macroactions = []


class RoadMapValueDict(MapValueDict):

    # format of files is assumed to be
    # loc_x, loc_y, demand, supp, n_count, n_1, ....n_{n_count}
    def __init__(self, filename):
        # TODO note hardcoded size of dataset
        self.dim_1 = 50
        self.dim_2 = 100

        # const for represanting that no data is available for this region
        self.NO_DATA_CONST = -1.0

        # because of the file format have to do some ugly parsing
        lines = open(filename).readlines()
        self.__number_of_points = len(lines)

        locs = np.empty((self.__number_of_points, 2))

        vals = np.empty((self.__number_of_points,))

        self.neighbours = {}

        # dict of selected macroactions
        self.selected_actions_dict = None

        # locations, where we have data
        self.informative_locations_indexes = []

        for i, line in enumerate(lines):
            a = StringIO(line)
            # get current point
            current_point = np.genfromtxt(a)

            current_loc = tuple(current_point[0:2])
            # cast neighbours to int list
            current_neighbours = map(int, current_point[4:].tolist())

            # the first item is count
            count = current_neighbours[0]

            # this list contains INTEGERS
            int_neighbours = current_neighbours[1:]
            assert len(int_neighbours) == count

            if count > 0:
                # x-1 because matlab numerates strings from 1, but locations are from 0
                tuple_neighbours = map(lambda x: LineToTuple(x, self.dim_1, self.dim_2), int_neighbours)
                self.neighbours[tuple(current_loc)] = tuple_neighbours

            raw_value = current_point[2]

            # TODO need to fix points with negative value
            adjusted_raw_value = raw_value + 1.0

            vals[i] = adjusted_raw_value if raw_value < 0 else raw_value

            if adjusted_raw_value != self.NO_DATA_CONST:
                self.informative_locations_indexes.append(i)

            # take only demand
            # vals[i] = self.NO_DATA_CONST if current_point[2] == self.NO_DATA_CONST else math.log(current_point[2] + 1.0)

            # copy location
            np.copyto(locs[i, :], current_loc)

        MapValueDict.__init__(self, locations=locs, values=vals)

        # TODO change mean so that it doesn't include self.NO_DATA locations
        self.mean = np.mean(vals[self.informative_locations_indexes])

    def GetNeighbours(self, location):
        tuple_loc = tuple(location)
        # int_neighbours = self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []
        # return map(lambda x: (x % self.dim_1, x / self.dim_1), int_neighbours)
        return self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []

    # list of 2D arrays
    def ___ExpandActions(self, start, batch_size):
        # including the start, hence +1
        if len(start) == batch_size + 1:
            yield np.asarray(start[1:])
        else:
            current = start[-1]

            for next_node in self.GetNeighbours(current):
                # Do we need the first condition?
                if (next_node in start) or (self.__call__(next_node) == self.NO_DATA_CONST):
                    continue
                # print self.__call__(next_node)
                for state in self.___ExpandActions(start + [next_node], batch_size):
                    yield state

    def SelectMacroActions(self, batch_size, folder_name, select_all=False):
        self.selected_actions_dict = {}

        treshhold = 20
        actions_file  = open(folder_name + 'actions_selected.txt', 'w')

        for loc in self.locations:
            all_macro_actions = self.GenerateAllRoadMacroActions(loc, batch_size)

            if select_all:
                self.selected_actions_dict[tuple(loc)] = all_macro_actions
                continue

            length = len(all_macro_actions)

            if length == 0:
                # do nothing
                continue
            elif length < treshhold:
                self.selected_actions_dict[tuple(loc)] = all_macro_actions
                actions_file.write(str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(range(length)) + '\n')
            else:
                generated_indexes = sample(xrange(length), treshhold)
                self.selected_actions_dict[tuple(loc)] = [all_macro_actions[i] for i in generated_indexes]
                actions_file.write(str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(generated_indexes) + '\n')

        actions_file.close()

    # for given state
    def GetSelectedMacroActions(self, current_state):
        current_state = tuple(current_state)
        return self.selected_actions_dict[current_state] if current_state in self.selected_actions_dict.keys() else []

    # for given state
    def GenerateAllRoadMacroActions(self, current_state, batch_size):
        current_state = tuple(current_state)
        return list(self.___ExpandActions([current_state], batch_size))

    def GetRandomStartLocation(self, batch_size):
        # now StartLocations point to all locations
        start_Locations = []
        for i in range(self.locations.shape[0]):
            loc = self.locations[i]

            # if we have at least one macroaction
            if self.GenerateAllRoadMacroActions(loc, batch_size) and self.__call__(loc) != self.NO_DATA_CONST:
                start_Locations.append(loc)
        return choice(start_Locations)

    # the content is moved to class constructor
    # unused
    def LogTransformValues(self):
        pass
        """
        for i in range(self.__number_of_points):
            current_value = self.values[i]
            if current_value != -1.0:

                self.values[i] = math.log(current_value + 1.0)
                # print current_value, self.values[i]
        """

    """
    def AddTwoSidedRoads(self):
        for loc in self.locations:
            tuple_loc = tuple(loc)
            for n in self.GetNeighbours(loc):
                # list of n's neighbours is empty
                n_neighbours = self.GetNeighbours(n)
                if not n_neighbours:
                    self.neighbours[tuple(n)] = [tuple_loc]
                else:
                    # list of n's neighbours is not empty, check if contains loc
                    if not tuple_loc in n_neighbours:
                        self.neighbours[tuple(n)].append(tuple_loc)
    """

    def LoadSelectedMacroactions(self, folder_name, batch_size):

        self.selected_actions_dict = {}

        actions_file_name = folder_name + 'actions_selected.txt'
        lines = open(actions_file_name).readlines()

        for line in lines:
            string_numbers = line.replace(',',' ').replace('[',' ').replace(']',' ').split()
            numbers = map(float, string_numbers)
            loc = ( numbers[0], numbers[1])
            indexes = map(int, numbers[2:])
            # print loc, indexes
            all_macro_actions = self.GenerateAllRoadMacroActions(loc, batch_size)
            length = len(indexes)
            assert length <= 20

            self.selected_actions_dict[tuple(loc)] = [all_macro_actions[i] for i in indexes]

    def LoadRandomLocation(self, folder_name):
        location_file_name = folder_name + 'start_location.txt'
        # should contain only one line
        string_locations = open(location_file_name).readline().split()
        location = map(float, string_locations)
        assert len(location) == 2
        return np.array(location)



if __name__ == "__main__":
    """
    filename = './taxi18.dom'
    # cannot use - cylcic linking
     m = GenerateRoadModelFromFile(filename)
    for i in m.locations:
        print i, m.GetNeighbours(i)
    """