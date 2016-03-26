import numpy as np
import sys
from HyperInference import __Cosines
from DatasetInfo import DropWaveInfo, AckleyInfo, CosinesInfo, BraninInfo, GriewankInfo, McCormickInfo, \
    SixCamelInfo, HolderTableInfo, __DatasetInfo
from TreePlanTester import TestScenario


# Random initial locations
def GenerateInitialLocation(current_f, batch_size):

    domain = current_f.domain
    gap = current_f.grid_gap
    min_x, max_x = domain[0]
    min_y, max_y = domain[1]

    # number of samples at each coordinate
    default_grid_size_x = float(max_x - min_x) / gap
    default_grid_size_y = float(max_y - min_y) / gap

    #todo
    # should add some check for this
    default_grid_size_x = int(default_grid_size_x)
    default_grid_size_y = int(default_grid_size_y)

    # in current implementation higher limit is not included hence -1
    int_x = np.random.random_integers(low=0, high=default_grid_size_x - 1, size=batch_size)
    int_y = np.random.random_integers(low=0, high=default_grid_size_y - 1, size=batch_size)

    # coordinates of location by x and by y
    int_x = min_x + gap * int_x
    int_y = min_y + gap * int_y

    return np.asarray([[int_x[i], int_y[i]] for i in range(batch_size)])


def GetBeta(iteration):
    beta_values = [0.0, 0.5, 1.0,  3.0, 5.0, 10.0]
    return beta_values[iteration]

def GetSimulatedFunction(i):
    if i == 0:
        return AckleyInfo()
    elif i ==1:
        return DropWaveInfo()
    elif i ==2:
        return CosinesInfo()

    elif i==3:
        return BraninInfo()
    elif i==4:
        return GriewankInfo()
    elif i==5:
        return McCormickInfo()
    elif i==6:
        return SixCamelInfo()
    else:
        return HolderTableInfo()

if __name__ == '__main__':
    
    args = sys.argv
    batch_size = 2
    save_trunk = "./tests/"

    # should be passed as params
    function_iteration = int(args[1])
    beta_iteration = int(args[2])
    location_iteration = int(args[3])

    #function_iteration = 2
    #beta_iteration = 1
    #location_iteration = 2
    current_function = GetSimulatedFunction(function_iteration)
    initial_location = GenerateInitialLocation(current_function, batch_size)
    beta = GetBeta(beta_iteration)

    print "function is " + str(current_function.name)
    print "beta is " + str(beta)
    print "location " + str(location_iteration) +  " is "  + str(initial_location)

    TestScenario(b=batch_size, beta=beta, location=initial_location, i = location_iteration,  simulated_func=current_function,
                 save_trunk=save_trunk)
    """
    test_f = __DatasetInfo(f=__Cosines, lengthscale=(0.12605123651, 0.126051232038),
                                   signal_variance=0.0198660061591, noise_variance=0.0001, mean=0.940527042428,
                                   domain=((0.0, 4.0), (0.0, 1.0)), grid_gap=0.05)
    for i in range(50):
        print GenerateInitialLocation(test_f,2)
    """