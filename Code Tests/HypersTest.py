import random

from src.DatasetUtils import GenerateRoadModelFromFile
from src.GaussianProcess import SquareExponential, GaussianProcess
from src.HypersStorer import RoadHypersStorer_Log44
import numpy as np
from math import sqrt

from src.Utils import GenerateGridPairs, TupleToLine


def TestPrediction(locs, vals):

    train_number_of_points = 10
    test_number_of_points = 5


    hypers = RoadHypersStorer_Log44()

    mu = hypers.mean_function

    # total number of points
    number_of_points = locs.shape[0]

    # choose a list of points from dataset
    indexes = random.sample(xrange(number_of_points), train_number_of_points+ test_number_of_points)

    # those are INDEXES
    train = indexes[:train_number_of_points]
    test = indexes[train_number_of_points:]
    print train, test

    # 2d array
    locs_train = locs[train, :]
    # print locs_train
    1# d array
    vals_train = vals[train]
    # print vals_train
    vals_train = np.atleast_2d(vals_train).T

    covariance_function = SquareExponential(length_scale=hypers.length_scale, signal_variance=hypers.signal_variance,
                                            noise_variance=hypers.noise_variance)

    gp = GaussianProcess(covariance_function, mean_function=mu)

    assert len(locs_train.shape) == 2
    print locs_train
    cholesky = gp.Cholesky(locs_train)

    error = 0
    for test_p in test:
        point = locs[test_p: test_p + 1, :]
        assert len(point.shape) == 2
        weights = gp.GPWeights(locations=locs_train, current_location=point, cholesky=cholesky)
        mu_predict = gp.GPMean(measurements=vals_train, weights=weights)

        truth = vals[test_p]
        print point, truth, mu_predict
        error += (mu_predict- truth)**2

    print sqrt(error / (test_number_of_points))


if __name__ == "__main__":
    file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(file_name)
    allowed_indexes = m.informative_locations_indexes
    locs = m.locations[allowed_indexes, :]
    vals = m.values[allowed_indexes]

    #TestPrediction(locs, vals)

    pairs = GenerateGridPairs(range(8,12), range(40,44)).tolist()

    raw_indexes = map(lambda x: TupleToLine(x, m.dim_1, m.dim_2), pairs)
    indexes = [x for x in raw_indexes if vals[x]!= m.NO_DATA_CONST]
    selected_locs = locs[indexes, :]
    print selected_locs
    selected_vals = vals[indexes]
    TestPrediction(selected_locs, selected_vals)