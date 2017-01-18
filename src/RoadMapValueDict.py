from StringIO import StringIO

from GaussianProcess import MapValueDict
import numpy as np

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
        number_of_points = len(lines)

        locs = np.empty((number_of_points, 2))

        vals = np.empty((number_of_points,))

        self.neighbours = {}

        for i, line in enumerate(lines):
            a = StringIO(line)
            # get current point
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
        int_neighbours =  self.neighbours[tuple_loc] if tuple_loc in self.neighbours.keys() else []
        return map(lambda x: np.array([ x % self.dim_1, x / self.dim_1]), int_neighbours)

    # UGLY
    # TODO change into generators
    def ExpandActions(self, start, batch_size):
        # including the start, hence +1
        if len(start) == batch_size + 1:
            # remove start state
            batch_road_macroactions.append(start[1:])
            return

        current = start[-1]
        for next_node in self.GetNeighbours(current):
            if next_node in start:
                continue
            self.ExpandActions(start + [next_node], batch_size)

    def GenerateRoadMacroActions(self, current_state, batch_size):
        self.ExpandActions([current_state], batch_size)
        return batch_road_macroactions
        # print batch_road_macroactions


if __name__ == "__main__":
    """
    filename = './taxi18.dom'
    # cannot use - cylcic linking
     m = GenerateRoadModelFromFile(filename)
    for i in m.locations:
        print i, m.GetNeighbours(i)
    """
