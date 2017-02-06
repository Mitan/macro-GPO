from StringIO import StringIO
from random import choice

from GaussianProcess import MapValueDict
import numpy as np
import math

from src.Utils import LineToTuple

batch_road_macroactions = []


class RoadMapValueDict(MapValueDict):

    # format of files is assumed to be
    # loc_x, loc_y, demand, supp, n_count, n_1, ....n_{n_count}
    def __init__(self, filename):
        # TODO note hardcoded size of dataset
        self.dim_1 = 50
        self.dim_2 = 100

        # const for represanting that no data is available for this region
        self.NO_DATA_CONST = -2.0

        # because of the file format have to do some ugly parsing
        lines = open(filename).readlines()
        self.__number_of_points = len(lines)

        locs = np.empty((self.__number_of_points, 2))

        vals = np.empty((self.__number_of_points,))

        self.neighbours = {}

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
            # vals[i] = raw_value + 1.0 if raw_value < 0 else raw_value
            vals[i] = raw_value

            if raw_value != self.NO_DATA_CONST:
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

    def GenerateRoadMacroActions(self, current_state, batch_size):
        current_state = tuple(current_state)
        return list(self.___ExpandActions([current_state], batch_size))

    def GetRandomStartLocation(self, batch_size):
        # now StartLocations point to all locations
        start_Locations = []
        for i in range(self.locations.shape[0]):
            loc = self.locations[i]

            # if we have at least one macroaction
            if self.GenerateRoadMacroActions(loc, batch_size) and self.__call__(loc) != self.NO_DATA_CONST:
                start_Locations.append(loc)
        return choice(start_Locations)

    # the content is moved to class constructor
    def LogTransformValues(self):
        pass
        """
        for i in range(self.__number_of_points):
            current_value = self.values[i]
            if current_value != -1.0:

                self.values[i] = math.log(current_value + 1.0)
                # print current_value, self.values[i]
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


if __name__ == "__main__":
    """
    filename = './taxi18.dom'
    # cannot use - cylcic linking
     m = GenerateRoadModelFromFile(filename)
    for i in m.locations:
        print i, m.GetNeighbours(i)
    """
