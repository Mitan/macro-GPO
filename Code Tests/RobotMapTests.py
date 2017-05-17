import scipy

from src.DatasetUtils import *
import numpy as np
import matplotlib.pyplot as plt


def IterateOverMacroActions(batch_size):
    data_file = '../datasets/robot/selected_slots/slot_2/final_slot_2.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'
    actions_folder = '../datasets/robot/'

    m = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                   neighbours_filename=neighbours_file)
    # m.AddTwoSidedRoads()
    locs = m.locations

    sum = 0.0
    max = 0
    max_loc = None
    count = 0
    for loc in locs:
        length = len(m.GenerateAllMacroActions(tuple(loc), batch_size))
        sum += length
        if length > max:
            max = length
            max_loc = loc

        # count the number of locations where we have macroactions
        if length > 0:
            # print loc, length
            count += 1
    print count, sum / count, max_loc, max

    print "Selecting macroactions"
    m.SelectMacroActions(batch_size=batch_size, folder_name = actions_folder)
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
    data_file = '../datasets/robot/selected_slots/slot_2/final_slot_2.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'
    actions_folder = '../datasets/robot/'

    m = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                   neighbours_filename=neighbours_file)

    start_location = np.array([m.GetRandomStartLocation(batch_size)])
    current_location = start_location[-1, :]
    print current_location

    print m.GenerateAllMacroActions(current_location, batch_size)


def GetMacroActionsOfLocation(loc, batch_size):
    data_file = '../datasets/robot/selected_slots/slot_2/final_slot_2.txt'
    neighbours_file = '../datasets/robot/all_neighbours.txt'
    coords_file = '../datasets/robot/all_coords.txt'

    m = GenerateRobotModelFromFile(data_filename=data_file, coords_filename=coords_file,
                                   neighbours_filename=neighbours_file)
    print m.GetNeighbours(loc)
    return m.GenerateAllMacroActions(loc, batch_size)


def LoadActionsFromFileTest(folder_name, batch_size):
    filename = '../datasets/slot18/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)
    m.LoadSelectedMacroactions(folder_name, batch_size)


if __name__ == '__main__':
    # IterateOverMacroActions(4)
    # GenerateMacroActionsFromeFile(4)
    l = [3.75,  23.5]
    for a in GetMacroActionsOfLocation(l, 4):
        print a
    # LogTransformTest()
    # HistTests()
    # LoadActionsFromFileTest('./', 4)
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
