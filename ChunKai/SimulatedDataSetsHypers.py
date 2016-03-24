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
    n = length * (length + 1) / 2.0
    mean_diff = sum_diff / n
    return 2.0 / mean_diff


# note - not working
def __PlotFunction(X, Y, Z):
    # X and Y are np arrays
    # ranges in our case
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


def GetSimulatedDataset(i, my_output_file):
    gridSizeX = 20
    gridSizeY = 20
    if i == 0:
        # Ackley
        X = np.arange(-5.0, 5.0, step= 0.5)
        Y = np.arange(-5.0, 5.0, step= 0.5)
        #Y = np.linspace(-15.0, 15.0, num=gridSizeY)
        f = __Ackley
        test_prediction_range = [-5.0, 5.0]

    elif i == 1:
        # DropWave
        X = np.arange(-1.0, 1.0, step= 0.1)
        Y = np.arange(-1.0, 1.0, step= 0.1)
        f = __DropWave
        test_prediction_range = [-1.0, 1.0]
    elif i == 2:
        # Griewank
        X = np.arange(-5.0, 5.0, step= 0.5)
        Y = np.arange(-5.0, 5.0, step= 0.5)
        f = _Griewank
        test_prediction_range = [-5.0, 5.0]

    elif i == 3:
        # Holder Table
        X = np.arange(-10.0, 10.0, step= 1.0)
        Y = np.arange(-10.0, 10.0, step= 1.0)
        #X = np.linspace(-10.0, 10.0, num=gridSizeX)
        #Y = np.linspace(-10.0, 10.0, num=gridSizeY)
        f = _HolderTable
        test_prediction_range = [-10.0, 10.0]

    elif i == 4:
        # Branin
        X = np.arange(-5.0, 10.0, step= 0.75)
        Y = np.arange(0.0, 15.0, step= 0.75)
        f = _Branin
        test_prediction_range = [0.0, 10.0]

    elif i == 5:
        # McCormick
        """
        X = np.linspace(-1.5, 4.0, num=gridSizeX)
        Y = np.linspace(-3.0, 4.0, num=gridSizeY)
        """
        X = np.arange(-1.0, 4.0, step= 0.25)
        Y = np.arange(-1.0, 4.0, step= 0.25)
        f = _McCormick
        test_prediction_range = [-1.5, 4.0]
    elif i == 6:
        # SixCamel
        X = np.arange(-2.0, 2.0, step= 0.2)
        Y = np.arange(-2.0, 2.0, step= 0.2)
        #X = np.linspace(-2.0, 2.0, num=gridSizeX)
        #Y = np.linspace(-2.0, 2.0, num=gridSizeY)
        f = _SixCamel
        test_prediction_range = [-2.0, 2.0]
    elif i == 7:
        # Shubert
        X = np.arange(-2.0, 2.0, step= 0.2)
        Y = np.arange(-2.0, 2.0, step= 0.2)
        f = __Shubert
        test_prediction_range = [-2.0, 2.0]
    elif i == 8:
        # Cosines
        X = np.arange(0.0, 1.0, step=0.05)
        Y = np.arange(0.0, 1.0, step=0.05)
        f = __Cosines
        test_prediction_range = [0.0, 1.0]
    elif i == 9:
        # Cosines
        X = np.linspace(-20.0, 20.0, num=gridSizeX)
        Y = np.linspace(-20.0, 20.0, num=gridSizeY)
        f = _Eggholder
        test_prediction_range = [-20.0, 20.0]
    else:
        raise ValueError("Imcorrect dataset number")
    print f
    grid = np.asarray([[x0, y0] for x0 in X for y0 in Y])
    values = np.apply_along_axis(f, 1, grid)
    values = values.reshape((values.shape[0], 1))

    # skew for checking if need to transform data
    skewness = skew(values)
    print skewness

    # __PlotFunction(X, Y, values)
    values_variance = np.std(values)

    # set initial values for parameters
    signal = values_variance
    # signal = __GetLengthScaleHeuristc(values)
    noise = signal * 0.1
    l_1 = __GetLengthScaleHeuristc(X)
    l_2 = __GetLengthScaleHeuristc(Y)

    # array for random restarts
    tests = [10 ** i for i in range(-3, 3)]

    m_best = None
    mu_best = 0.0
    likeilhood_best = - np.Inf

    # train hypers with several random starts
    for i in range(len(tests)):
        m, mu = InferHypers(grid, values, noise * tests[i], signal * tests[i], l_1 * tests[i - 1], l_2 * tests[i - 1])
        if m.likelihood > likeilhood_best:
            m_best = m
            mu_best = mu
            likeilhood_best = m.likelihood
    print m_best
    mse = TestPrediction(m_best, mu_best, f, test_prediction_range)
    #print "MSE is " + str(mse) + " with data variance " + str(values_variance)
    WriteInfoToFile(my_output_file, f, mse, values_variance, m, mu)

    return grid, values


def WriteInfoToFile(my_output_file,f, mse, variance, m, mu):
    my_output_file.write("function is " + str(f) + "\n")
    my_output_file.write("MSE is " + str(mse) + " with data variance " + str(variance)+ "\n")

    #todo note need to square the l_1 and l_2
    l_1, l_2 =  m.param_array[1:3]
    my_output_file.write("l_1 = " + str(l_1) + "\n")
    my_output_file.write("l_2 = " + str(l_2) + "\n")

    # todo note this is already sigma^2
    noise_variance = m.param_array[3]
    my_output_file.write("sigma noise = " + str(noise_variance) + "\n")

    signal_variance = m.param_array[0]
    my_output_file.write("sigma signal = " + str(signal_variance) + "\n")

    my_output_file.write("mean = " + str(mu) + "\n\n")


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
    x = np.asarray(x)
    assert x.shape[0] == 2
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    # todo note dimension is hardcoded
    d = 2
    s_1 = np.sum(np.square(x))
    s_2 = np.sum(np.cos(c * x))
    term1 = -a * math.exp(-b * math.sqrt(s_1 / d))
    term2 = -math.exp(s_2 / d)
    y = term1 + term2 + a + math.exp(1.0)
    # Warning!
    # real Ackley function is y
    # but to ensure small skewness we perform transformation
    return math.log(-y + 21.3)


def __DropWave(x):
    # return Dropwave function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    frac1 = 1 + math.cos(12 * math.sqrt(x1 ** 2 + x2 ** 2))
    frac2 = 0.5 * (x1 ** 2 + x2 ** 2) + 2
    y = frac1 / frac2
    # return y
    return y


def _Griewank(x):
    # return Griewank function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    sum = (x1 ** 2 + x2 ** 2) / 4000.0
    prod = math.cos(x1) * math.cos(x2 / math.sqrt(2))
    y = sum - prod + 1
    return y

# could be better
def _HolderTable(x):
    # return HolderTable function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    fact1 = math.sin(x1) * math.cos(x2)
    fact2 = math.exp(abs(1 - math.sqrt(x1 ** 2 + x2 ** 2) / math.pi))
    y = -abs(fact1 * fact2)
    return math.log(-y + 0.25)

# Note, doesn't work - exploartion matrix det is zero
def __Shubert(x):
    # return HolderTable function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    list_x1 = [i * math.cos((i + 1) * x1 + i) for i in range(1, 6)]
    list_x2 = [i * math.cos((i + 1) * x2 + i) for i in range(1, 6)]
    y = sum(list_x1) * sum(list_x2)
    return y


def _Branin(x):
    # return Branin function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)
    term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * math.cos(x1)
    y = term1 + term2 + s
    return math.log(y + 5.0)


def _McCormick(x):
    # return Branin function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    term1 = math.sin(x1 + x2)
    term2 = (x1 - x2) ** 2
    term3 = -1.5 * x1
    term4 = 2.5 * x2
    y = term1 + term2 + term3 + term4 + 1
    return math.log(y + 2.5)


def _SixCamel(x):
    # return Branin function
    # this function is 2D
    x = np.asarray(x)
    assert x.shape[0] == 2
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    y = term1 + term2 + term3
    return math.log(y + 1.2)


def __Cosines(x):
    x = np.asarray(x)
    assert x.shape[0] == 2
    cosine_list = [__g_cosines(x[i]) - __r_cosines(x[i]) for i in range(2)]
    y = 1 - sum(cosine_list)
    return math.log(-y + 3.)


def __g_cosines(x):
    return (1.6 * x - 0.5) ** 2


def __r_cosines(x):
    return 0.3 * math.cos(3 * math.pi * (1.6 * x - 0.5))


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



"""
# Note, POSSIBLY doesn't work - exploartion matrix det is zero
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




def TestPrediction(m, mu, f, preditcion_range):
    # test points
    n = 50
    grid = np.random.uniform(preditcion_range[0], preditcion_range[1], (n, 2))
    # print at
    predictions = [(m.predict(np.atleast_2d(grid[i, :]))[0])[0, 0] for i in range(n)]
    # model m uses zero mean, so need to add it
    predictions = mu + np.asarray(predictions)
    ground_truth = np.apply_along_axis(f, 1, grid)
    diff = predictions - ground_truth
    squares = diff * diff
    assert diff.shape == squares.shape
    mse = np.sum(squares) / n
    return mse


if __name__ == "__main__":
    my_file =  open("./datasets/simulated-functions-hypers_test.txt", 'w')
    t = 3
    for i in range(t,t+1):
        GetSimulatedDataset(i, my_file)
    my_file.close()
