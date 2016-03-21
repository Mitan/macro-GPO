import math
import numpy as np
from hypers import InferHypers


def GetSimulatedDataset(i):
    if i==0:
        grid, values, bad_places = __GetAckleyValues()
        hypers = InferHypers(grid, values, 0.1, 3.0, 1.0, 1.0)
    else:
        raise ValueError("Imcorrect dataset number")
    return hypers, grid, values, bad_places

x = np.array([1.0, 2.0])

def __Ackley(x):
    # return Ackley function multiplied by -1
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    #todo note dimension is hardcoded
    d  = x.shape[0]
    s_1 = np.sum(np.square(x))
    s_2 = np.sum(np.cos(c*x))
    term1  = -a * math.exp(-b*math.sqrt(s_1/d))
    term2 = -math.exp(s_2/d)
    y =  term1 + term2 + a + math.exp(1.0)
    return -y
#print __Ackley(x)

def __GetAckleyValues():
    grid_range = 15.0
    X = np.arange(-grid_range, grid_range+1)
    Y = np.arange(-grid_range, grid_range+1)
    grid =np.asarray([[x0, y0] for x0 in X for y0 in Y])
    #v_Ackley = np.vectorize(__Ackley)
    values = np.apply_along_axis(__Ackley, 1, grid)
    values = values.reshape((values.shape[0],1))
    # return grid, values and bad places
    return grid, values, None

print GetSimulatedDataset(0)