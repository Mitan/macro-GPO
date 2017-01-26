from src.DatasetUtils import GenerateRoadModelFromFile
import numpy as np


def IterateOverMacroActions(batch_size):
    file_name = file_name = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)
    m.AddTwoSidedRoads()
    locs = m.locations

    max = 0
    count = 0
    for loc in locs:
        length = len(m.GenerateRoadMacroActions(tuple(loc), batch_size))

        if length > max:
            max = length

        if length > 0:
            # print loc, length
            count += 1
    print count, max


def GenerateMacroActionsFromeFile(batch_size):
    filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)

    start_location = np.array([m.GetRandomStartLocation(batch_size)])
    current_location = start_location[-1, :]
    print current_location

    print m.GenerateRoadMacroActions(current_location, batch_size)[0].shape


def GetMacroActionsOfLocation(loc, batch_size):
    filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)
    return m.GenerateRoadMacroActions(loc, batch_size)


def CheckNeighboursTest():
    file_name = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)

    locs = m.locations

    for loc in locs:
        loc = tuple(loc)
        # length = len(m.GenerateRoadMacroActions(tuple(loc), 3))
        neighbours = m.GetNeighbours(loc)
        # print loc, neighbours
        for nei in neighbours:
            n_neighbours = m.GetNeighbours(nei)

            if loc in n_neighbours:
                print loc, nei


def LogTransformTest():
    file_name = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)
    vals = m.values
    for val in vals:
        print val


if __name__ == '__main__':
    IterateOverMacroActions(4)
    # GenerateMacroActionsFromeFile()
    # CheckNeighboursTest()
    # print GetMacroActionsOfLocation([29.0, 6.0], 4)
    # LogTransformTest()

    """
    filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)
    locs = m.locations

    old_neighbours = {}
    for loc in locs:
            old_neighbours[tuple(loc)] = len(m.GetNeighbours(loc))
    loc0 = [8.0, 56.0]
    # print loc0
    for loc in locs:
        tuple_loc = tuple(loc)
        for n in m.GetNeighbours(loc):
            # list of n's neighbours is empty
            n_neighbours = m.GetNeighbours(n)
            if not n_neighbours:
                m.neighbours[tuple(n)] = [tuple_loc]
            else:
                # list of n's neighbours is not empty, check if contains loc
                if not tuple_loc in n_neighbours:
                    m.neighbours[tuple(n)].append(tuple_loc)

    for loc in locs:
        print loc, old_neighbours[tuple(loc)], len(m.GetNeighbours(loc))
    """