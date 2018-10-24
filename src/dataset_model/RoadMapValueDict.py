from StringIO import StringIO
from random import choice, sample

from src.dataset_model.MapValueDictBase import MapValueDictBase
import numpy as np

from src.Utils import LineToTuple

from src.enum.DatasetEnum import DatasetEnum

batch_road_macroactions = []


class RoadMapValueDict(MapValueDictBase):

    # format of files is assumed to be
    # loc_x, loc_y, demand, supp, n_count, n_1, ....n_{n_count}
    def __init__(self, filename, hyper_storer, domain_descriptor, batch_size):
        self.dataset_type = DatasetEnum.Road
        self.hyper_storer = hyper_storer
        self.domain_descriptor = domain_descriptor
        self.batch_size = batch_size

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

        MapValueDictBase.__init__(self, locations=locs, values=vals)

        # TODO change mean so that it doesn't include self.NO_DATA locations
        self.mean = np.mean(vals[self.informative_locations_indexes])

    def GetNeighbours(self, location):
        tuple_loc = tuple(location)
        # int_neighbours = self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []
        # return map(lambda x: (x % self.dim_1, x / self.dim_1), int_neighbours)
        return self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []

    # list of 2D arrays
    def ___ExpandActions(self, start):
        # including the start, hence +1
        if len(start) == self.batch_size + 1:
            yield np.asarray(start[1:])
        else:
            current = start[-1]

            for next_node in self.GetNeighbours(current):
                # Do we need the first condition?
                if (next_node in start) or (self.__call__(next_node) == self.NO_DATA_CONST):
                    continue
                # print self.__call__(next_node)
                for state in self.___ExpandActions(start + [next_node]):
                    yield state

    def SelectMacroActions(self, actions_filename, ma_treshold):
        self.selected_actions_dict = {}

        actions_file  = open(actions_filename, 'w') if ma_treshold else None

        for loc in self.locations:
            all_macro_actions = self.GenerateAllRoadMacroActions(loc)

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
    def GenerateAllRoadMacroActions(self, current_state):
        current_state = tuple(current_state)
        return list(self.___ExpandActions([current_state]))

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

    def LoadSelectedMacroactions(self, actions_filename):

        self.selected_actions_dict = {}

        lines = open(actions_filename).readlines()

        for line in lines:
            string_numbers = line.replace(',',' ').replace('[',' ').replace(']',' ').split()
            numbers = map(float, string_numbers)
            loc = ( numbers[0], numbers[1])
            indexes = map(int, numbers[2:])
            # print loc, indexes
            all_macro_actions = self.GenerateAllRoadMacroActions(loc)
            length = len(indexes)
            assert length <= 20

            self.selected_actions_dict[tuple(loc)] = [all_macro_actions[i] for i in indexes]

    def GenerateStartLocation(self):
        # now StartLocations point to all locations
        start_Locations = []
        for i in range(self.locations.shape[0]):
            loc = self.locations[i]

            # if we have at least one macroaction
            if self.GenerateAllRoadMacroActions(loc) and self.__call__(loc) != self.NO_DATA_CONST:
                start_Locations.append(loc)
        self.start_location = np.array([choice(start_Locations)])

    def LoadStartLocation(self, location_filename):
        # should contain only one line
        string_locations = open(location_filename).readline().split()
        location = map(float, string_locations)
        assert len(location) == 2
        self.start_location = np.array([location])
