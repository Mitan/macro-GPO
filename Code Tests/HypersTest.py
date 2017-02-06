from src.DatasetUtils import GenerateRoadModelFromFile
from src.GaussianProcess import SquareExponential, GaussianProcess
from src.HypersStorer import RoadHypersStorer_Log44
import numpy as np
from math import sqrt


def TestPrediction(locs, vals):

    test_num_point = 20

    hypers = RoadHypersStorer_Log44()

    mu = hypers.mean_function
    vals = vals - mu

    # total number of points
    number_of_points = locs.shape[0]
    array_range = np.arange(number_of_points)

    # choose a list of points from dataset
    indexes = np.random.choice(array_range, test_num_point).tolist()

    half_points = test_num_point / 2
    train = indexes[:half_points]
    test = indexes[half_points:]
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
        # print truth, mu_predict
        error += (mu_predict- truth)**2

    print sqrt(error / (half_points))


if __name__ == "__main__":
    file_name = '../datasets/slot18/tlog18.dom'
    m = GenerateRoadModelFromFile(file_name)
    allowed_indexes = m.informative_locations_indexes
    locs = m.locations[allowed_indexes, :]
    vals = m.values[allowed_indexes]

    #TestPrediction(locs, vals)

    g = np.meshgrid(range(10, 12), range(110,114))
    pairs = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    print pairs