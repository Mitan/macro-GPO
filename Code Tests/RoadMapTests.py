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


def GenerateMacroActionsFromeFile():
    filename = filename = '../src/taxi18.dom'
    m = GenerateRoadModelFromFile(filename)
    batch_size = 3
    start_location = np.array([m.GetRandomStartLocation(batch_size)])
    current_location = start_location[-1, :]
    print current_location

    print m.GenerateRoadMacroActions(current_location, batch_size)[0].shape


if __name__=='__main__':
    # IterateOverMacroActions()
    GenerateMacroActionsFromeFile()