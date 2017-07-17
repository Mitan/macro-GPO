import math
import numpy as np
import scipy

from src.GaussianProcess import SquareExponential, GaussianProcess

def LeaveOneOut(slot_number):
    error = 0
    input_data_file = '../unused_datasets/old_robot/slots/slot_' + str(slot_number) + '.txt'
    hypers_file = '../unused_datasets/old_robot/hypers/hypers_' + str(slot_number) + '.txt'
    all_data = np.genfromtxt(input_data_file)
    hypers = np.genfromtxt(hypers_file)
    X = all_data[:, :-1]
    Y = all_data[:, -1:]
    num_points, point_dimension = X.shape
    mean = hypers[0]
    length_scale = hypers[1: -2]
    assert length_scale.shape[0] == point_dimension

    signal_variance = hypers[-2]
    noise_variance = hypers[-1]

    covariance_function = SquareExponential(length_scale, signal_variance, noise_variance)
    gp = GaussianProcess(covariance_function=covariance_function, mean_function=mean)

    for index in range(num_points):
        train_indexes = [i for i in range(num_points) if i != index]
        train_X = X[train_indexes, :]
        train_Y = Y[train_indexes, :]
        test_location = X[index: index + 1, :]

        # predict
        cholesky = gp.Cholesky(train_X)
        weights = gp.GPWeights(locations=train_X, current_location=test_location, cholesky=cholesky)
        mu = gp.GPMean(measurements=train_Y, weights=weights)[0, 0]
        error += (mu - Y[index, 0]) ** 2
    # return math.sqrt(error / num_points), scipy.stats.skew(Y)[0], np.std(Y)
    return math.sqrt(error / num_points) / np.std(Y), scipy.stats.skew(Y)[0]

if __name__ == "__main__":
    for k in range(25):
        print k, LeaveOneOut(slot_number=k)


