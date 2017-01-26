from StringIO import StringIO
from random import choice

from GaussianProcess import MapValueDict
import numpy as np
import math

batch_road_macroactions = []


class RoadMapValueDict(MapValueDict):


    # format of files is assumed to be
    # loc_x, loc_y, demand, supp, n_count, n_1, ....n_{n_count}
    def __init__(self, filename):
        # TODO note hardcoded size of dataset
        self.dim_1 = 50
        self.dim_2 = 100

        # because of the file format have to do some ugly parsing
        lines = open(filename).readlines()
        self.__number_of_points = len(lines)

        locs = np.empty((self.__number_of_points, 2))

        vals = np.empty((self.__number_of_points,))

        self.neighbours = {}

        for i, line in enumerate(lines):
            a = StringIO(line)
            # get current point
            current_point = np.genfromtxt(a)

            current_loc = tuple(current_point[0:2])
            current_neighbours = map(int, current_point[4:].tolist())

            # the first item is count
            count = current_neighbours[0]
            assert len(current_neighbours[1:]) == count

            if count > 0:
                self.neighbours[tuple(current_loc)] = current_neighbours[1:]

            # take only demand
            vals[i] = current_point[2]
            # copy location
            np.copyto(locs[i, :], current_loc)

        MapValueDict.__init__(self, locations=locs, values=vals)

    def GetNeighbours(self, location):
        tuple_loc = tuple(location)
        int_neighbours =  self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []
        return map(lambda x: (x % self.dim_1, x / self.dim_1), int_neighbours)

    def ___ExpandActions(self, start, batch_size):
        # including the start, hence +1
        if len(start) == batch_size + 1:
            yield np.asarray(start[1:])
        else:
            current = start[-1]

            for next_node in self.GetNeighbours(current):
                # Do we need the first condition?
                if (next_node in start) or (self.__call__(next_node) == -1.0):
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
            if self.GenerateRoadMacroActions(loc, batch_size):
                start_Locations.append(loc)
        return choice(start_Locations)

    def LogTransformValues(self):
        for i in range(self.__number_of_points):
            current_value = self.values[i]
            if current_value != -1.0:
                self.values[i] = math.log(current_value + 1.0)

    def AddTwoSidedRoads(self):
        pass


if __name__ == "__main__":
    """
    filename = './taxi18.dom'
    # cannot use - cylcic linking
     m = GenerateRoadModelFromFile(filename)
    for i in m.locations:
        print i, m.GetNeighbours(i)
    """
