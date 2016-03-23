import math

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

from hypers import InferHypers


def __GetLengthScaleHeuristc(inputs):
    sum_diff = 0.0
    length = inputs.shape[0]
    for i in range(length):
        for j in range(i, length):
            sum_diff += inputs[j] - inputs[i]
    n = length * (length+1) / 2.0
    mean_diff = sum_diff / n
    return 2.0 / mean_diff

#note - not working
def __PlotFunction(X, Y, Z):
    # X and Y are np arrays
    # ranges in our case
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()

def __CreateGrid(x_max, x_min, y_max, y_min):
    X = np.arange(-x_min, x_max + 1)
    Y = np.arange(-y_min, y_max + 1)
    grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
    return grid

def GetSimulatedDataset(i):
    bad_places = []
    if i==0:
        # Ackley
        X = np.linspace(-15.0, 15.0, num=20)
        Y = np.linspace(-15.0, 15.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(__Ackley, 1, grid)

    elif i==1:
        #DropWave
        X = np.linspace(-1.0, 1.0, num=20)
        Y = np.linspace(-1.0, 1.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(__DropWave, 1, grid)
    elif i ==2:
        # Griewank
        X = np.linspace(-5.0, 5.0, num=20)
        Y = np.linspace(-5.0, 5.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_Griewank, 1, grid)
    elif i ==3:
        # Holder Table
        X = np.linspace(-10.0, 10.0, num=20)
        Y = np.linspace(-10.0, 10.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_HolderTable, 1, grid)

    elif i ==4:
        # Branin
        X = np.linspace(-5.0, 10.0, num=20)
        Y = np.linspace(0.0, 15.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_Branin, 1, grid)

    elif i ==5:
        # McCormick
        X = np.linspace(-1.5, 4.0, num=20)
        Y = np.linspace(-3.0, 4.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_McCormick, 1, grid)
    elif i ==6:
        # SixCamel
        X = np.linspace(-3.0, 3.0, num=20)
        Y = np.linspace(-2.0, 2.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_SixCamel, 1, grid)
    elif i ==7:
        # Shubert
        X = np.linspace(-2.0, 2.0, num=20)
        Y = np.linspace(-2.0, 2.0, num=20)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(__Shubert, 1, grid)

    else:
        raise ValueError("Imcorrect dataset number")

    values = values.reshape((values.shape[0],1))
    skewness = skew(values)
    print skewness
    #__PlotFunction(X, Y, values)
    signal = np.std(values)
    #signal = __GetLengthScaleHeuristc(values)
    noise = signal * 0.1
    l_1 = __GetLengthScaleHeuristc(X)
    l_2 = __GetLengthScaleHeuristc(Y)
    tests = [10**i for i in range(-3,3)]
    for i in range(len(tests)):
        hypers = InferHypers(grid, values, noise * tests[i], signal * tests[i], l_1 * tests[i-1], l_2 * tests[i-1])
    return hypers, grid, values, bad_places


"""
    elif i==4:
        #Eggholder
        # should be 512
        d = 20.0
        X = np.arange(-d, d+1)
        Y = np.arange(-d, d+1)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(_Eggholder, 1, grid)

    elif i==1:
        #Bukin6
        X = np.arange(-5.0, 6.0)
        Y = np.arange(-3.0, 4.0)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(__Bukin6, 1, grid)

    elif i==2:
        #CrossInTray
        X = np.arange(-10.0, 10.0+1)
        Y = np.arange(-10.0, 10.0+1)
        grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
        values = np.apply_along_axis(__CrosInTray, 1, grid)
"""

def __Ackley(x):
    # return Ackley function
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    #todo note dimension is hardcoded
    d  = 2
    s_1 = np.sum(np.square(x))
    s_2 = np.sum(np.cos(c*x))
    term1  = -a * math.exp(-b*math.sqrt(s_1/d))
    term2 = -math.exp(s_2/d)
    y =  term1 + term2 + a + math.exp(1.0)
    return y

def __DropWave(x):
    # return Dropwave function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    frac1 =  1 + math.cos(12*math.sqrt(x1**2+x2**2))
    frac2 = 0.5*(x1**2+x2**2) + 2
    y = frac1/frac2
    #return y
    return y

def _Griewank(x):
    # return Griewank function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    sum = (x1**2 + x2**2) / 4000.0
    prod = math.cos(x1) * math.cos(x2 / math.sqrt(2))
    y = sum - prod + 1
    return y

def _HolderTable(x):
    # return HolderTable function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    fact1 =  math.sin(x1)*math.cos(x2)
    fact2 = math.exp(abs(1 - math.sqrt(x1**2+x2**2)/math.pi))
    y = -abs(fact1*fact2)
    return y


def __Shubert(x):
    # return HolderTable function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    list_x1 = [i * math.cos((i+1)* x1 + i) for i in range(1,6)]
    list_x2 = [i * math.cos((i+1)* x2 + i) for i in range(1,6)]
    y = sum(list_x1) * sum(list_x2)
    return y

def _Branin(x):
    # return Branin function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1/(4* math.pi**2)
    c =  5.0/math.pi
    r = 6.0
    s = 10.0
    t = 1.0 /(8 * math.pi)
    term1 =  a * (x2 - b* x1**2 + c * x1 - r)**2
    term2  =  s * (1-t) * math.cos(x1)
    y =  term1 + term2 + s
    return y

def _McCormick(x):
    # return Branin function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    term1 = math.sin(x1 + x2)
    term2  = (x1 - x2)**2
    term3 = -1.5 * x1
    term4 = 2.5 * x2
    y = term1 + term2 + term3 + term4 + 1
    return y

def _SixCamel(x):
    # return Branin function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    term1 =  (4-2.1 * x1**2 +  (x1**4) / 3 )  * x1**2
    term2 = x1 * x2
    term3 =  (-4 + 4 * x2**2) * x2**2
    y = term1 + term2 + term3
    return y
"""
def __Bukin6(x):
    # return Bukin function n 6
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    term1 =  100 * math.sqrt(abs(x2 - 0.01*(x1)**2))
    term2 = 0.01 * abs(x1+10)
    y = term1 + term2
    return y

def __CrosInTray(x):
    # return CrosInTray function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    fact1 = math.sin(x1)*math.sin(x2)
    fact2 = math.exp(abs(100 - math.sqrt(x1**2+x2**2)/math.pi))
    y = -0.0001 * (abs(fact1*fact2)+1)**0.1
    return y




def _Eggholder(x):
    # return EGGHOLDER function
    #this function is 2D
    assert x.shape[0] ==2
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2+47) * math.sin(math.sqrt(abs(x2 + x1/2.0 + 47)))
    term2 = -x1 * math.sin(math.sqrt(abs(x1 - (x2 + 47))))
    y = term1 + term2
    return y
    #return math.log(-y + 0.02)

"""



if __name__ == "__main__":
    GetSimulatedDataset(7)