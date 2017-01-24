from src.DatasetUtils import GenerateRoadModelFromFile
import numpy as np


def IterateOverMacroActions():
    file_name = file_name = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(file_name)

    locs = m.locations

    count = 0
    for loc in locs:
        length = len(m.GenerateRoadMacroActions(tuple(loc), 3))
        if length > 0:
            print loc, length
            count+=1
    print count


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

if __name__=='__main__':
    # IterateOverMacroActions()
    # GenerateMacroActionsFromeFile()
    # CheckNeighboursTest()
    print GetMacroActionsOfLocation([29.0, 6.0], 4)