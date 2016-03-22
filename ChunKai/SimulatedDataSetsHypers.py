import math
import numpy as np
from scipy.stats import skew
from hypers import InferHypers
import matplotlib.pyplot as plt


def GetSimulatedDataset(i):
    if i==0:
        #Ackley
        grid, values, bad_places = __GetAckleyValues()
        print skew(values)
        hypers = InferHypers(grid, values, 0.05, 0.5, 1.0, 1.0)
    elif i==1:
        #Bukin6
        grid, values, bad_places = __GetBukin6Values()
        print skew(values)

        n, bins, patches = plt.hist(values, 40, normed=1, facecolor='green', alpha=0.75)
        plt.show()
        #hypers = 0
        hypers = InferHypers(grid, values, 0.04, 0.4, 3.0, 3.0)
    elif i==2:
        #DropWave
        grid, values, bad_places = __GetDropWaveValues()
        hypers = InferHypers(grid, values, 0.05, 0.5, 1.0, 1.0)
    else:
        raise ValueError("Imcorrect dataset number")
    print hypers
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

# bad example to fit GP in it

def __Bukin6(x):
    # return Bukin function n 6 multiplied by -1
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    term1 =  100 * math.sqrt(abs(x2 - 0.01*(x1)**2))
    term2 = 0.01 * abs(x1+10)
    y = term1 + term2
    return -y

def __GetBukin6Values():
    X = np.arange(-5.0, 6.0)
    Y = np.arange(-3.0, 4.0)
    grid =np.asarray([[x0, y0] for x0 in X for y0 in Y])
    values = np.apply_along_axis(__Bukin6, 1, grid)
    values = values.reshape((values.shape[0],1))
    # return grid, values and bad places
    return grid, values, None


def __CrosInTray(x):
    # return CrosInTray function  multiplied by -1
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    fact1 = math.sin(x1)*math.sin(x2)
    fact2 = math.exp(abs(100 - math.sqrt(x1**2+x2**2)/math.pi))
    y = -0.0001 * (abs(fact1*fact2)+1)**0.1
    return -y

def __GetCrosInTrayValues():
    grid_range = 10.0
    X = np.arange(-grid_range, grid_range+1)
    Y = np.arange(-grid_range, grid_range+1)
    grid =np.asarray([[x0, y0] for x0 in X for y0 in Y])
    values = np.apply_along_axis(__CrosInTray, 1, grid)
    values = values.reshape((values.shape[0],1))
    # return grid, values and bad places
    return grid, values, None


def __DropWave(x):
    # return CrosInTray function  multiplied by 1
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    frac1 =  1 + math.cos(12*math.sqrt(x1**2+x2**2))
    frac2 = 0.5*(x1**2+x2**2) + 2
    y = frac1/frac2
    return math.log(y + 0.02)
    #return math.log(-y + 0.02)

def __GetDropWaveValues():
    grid_range = 5.0
    step = 0.5
    X = np.arange(-grid_range, grid_range+1, step= step)
    Y = np.arange(-grid_range, grid_range+1, step= step)
    grid =np.asarray([[x0, y0] for x0 in X for y0 in Y])
    values = np.apply_along_axis(__DropWave, 1, grid)
    values = values.reshape((values.shape[0],1))
    # return grid, values and bad places
    return grid, values, None

if __name__ == "__main__":
    GetSimulatedDataset(0)