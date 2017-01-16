from StringIO import StringIO

from GaussianProcess import MapValueDict
import numpy as np


class RoadMapValueDict(MapValueDict):


    # format of files is assumed to be
    # loc_x, loc_y, demand, supp, n_count, n_1, ....n_{n_count}
    def __init__(self, filename):
        # because of the file format have to do some ugly parsing
        lines = open(filename).readlines()
        number_of_points = len(lines)

        locs = np.empty((number_of_points, 2))

        vals = np.empty((number_of_points,))

        self.neighbours = {}

        for i, line in enumerate(lines):
            a = StringIO(line)
            current_point = np.genfromtxt(a)

            current_loc = current_point[0:2]
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
        return self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []

