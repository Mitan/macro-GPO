import numpy as np


def TupleToLine(tuple_location, dim_1, dim_2):
    float_line =  tuple_location[0] * dim_2 + tuple_location[1] + 1
    return int(float_line)


def LineToTuple(line_location, dim_1, dim_2):
    return (  float((line_location - 1) / dim_2),   float((line_location - 1) % dim_2)  )

# arguments are lists
def GenerateGridPairs(first_range, second_range):
    g = np.meshgrid(first_range, second_range)
    pairs = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    return pairs


# converts ndarray to tuple
# can't pass ndarray as a key for dict
def ToTuple(arr):
    return tuple(map(tuple, arr))