import scipy

from src.DatasetUtils import GenerateRoadModelFromFile
import numpy as np
import matplotlib.pyplot as plt


def IterateOverMacroActions(batch_size):
    file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(file_name)
    # m.AddTwoSidedRoads()
    locs = m.locations

    sum = 0.0
    max = 0
    count = 0
    for loc in locs:
        length = len(m.GenerateAllRoadMacroActions(tuple(loc), batch_size))
        sum+= length
        if length > max:
            max = length

        # count the number of locations where we have macroactions
        if length > 0:
            # print loc, length
            count += 1
    print count, max, sum / count

    print "Selecting macroactions"
    m.SelectMacroActions(batch_size)
    sum = 0.0
    max = 0
    count = 0

    for loc in locs:
        length = len(m.GetSelectedMacroActions(tuple(loc)))
        sum+= length
        if length > max:
            max = length
        # count the number of locations where we have macroactions
        if length > 0:
            # print loc, length
            count += 1
    print count, max, sum / count




def GenerateMacroActionsFromeFile(batch_size):
    filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)

    start_location = np.array([m.GetRandomStartLocation(batch_size)])
    current_location = start_location[-1, :]
    print current_location

    print m.GenerateAllRoadMacroActions(current_location, batch_size)[0].shape


def GetMacroActionsOfLocation(loc, batch_size):
    filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)
    return m.GenerateAllRoadMacroActions(loc, batch_size)


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


def NewDatasetTest():
    file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(file_name)
    print m.mean

def LogTransformTest():
    file_name = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)
    vals = m.values
    for val in vals:
        print val

def HistTests():
    file_name = '../src/taxi44.dom'
    m = GenerateRoadModelFromFile(file_name)

    hypers_file = open('hypers44.txt', 'w')
    locs = m.locations
    vals = m.values
    for i in range(locs.shape[0]):
        if vals[i] != -1.0:
            loc = locs[i, :]
            hypers_file.write(str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(vals[i]) + '\n')
    hypers_file.close()

    list_vals = [v for v in vals.tolist() if v!=-1.0]
    updated_vals = np.asarray(list_vals)
    print scipy.stats.skew(updated_vals)
    plt.hist(updated_vals, bins=50)

    plt.show()

if __name__ == '__main__':
    IterateOverMacroActions(4)
    # NewDatasetTest()
    # GenerateMacroActionsFromeFile()
    # CheckNeighboursTest()
    # print GetMacroActionsOfLocation([29.0, 6.0], 4)
    # LogTransformTest()
    # HistTests()
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

