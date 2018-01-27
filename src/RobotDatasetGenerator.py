import numpy as np
from scipy.stats import multivariate_normal

from src.GaussianProcess import SquareExponential, GaussianProcess


# exact values based on GP prediction
def GenerateRandomRobotDataset(slot_number, seed):
    input_data_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/slot_' + str(slot_number) + '.txt'
    hypers_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/hypers_' + str(slot_number) + '.txt'

    generated_coords_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/generated_id_coords_slot_' + str(
        slot_number) + '.txt'

    all_generated_coords_id = np.genfromtxt(generated_coords_file)
    generated_cords = all_generated_coords_id[:, 1:]

    all_data = np.genfromtxt(input_data_file)
    hypers = np.genfromtxt(hypers_file)

    #  first row is id
    X = all_data[:, 1:-1]
    Y = all_data[:, -1:]

    num_points, point_dimension = X.shape
    mean = hypers[0]
    length_scale = hypers[1: -2]
    assert length_scale.shape[0] == point_dimension

    signal_variance = hypers[-2]
    noise_variance = hypers[-1]

    covariance_function = SquareExponential(length_scale, signal_variance, noise_variance)
    gp = GaussianProcess(covariance_function=covariance_function, mean_function=mean)
    mu, var = __predictGP(gp=gp, train_X=X, train_Y=Y, test_location=generated_cords)

    np.random.seed(seed=seed)
    # print mu[:, 0].shape
    drawn_vector = multivariate_normal.rvs(mean=mu[:, 0], cov=var)
    rounding_f = np.vectorize(lambda x: round(x, 4))
    drawn_vector = rounding_f(drawn_vector)
    """
    for i in range(104):
        print mu[i, 0], drawn_vector[i]
    """
    # print all_generated_coords_id.shape, np.atleast_2d(drawn_vector).T.shape
    generated_data = np.concatenate((all_generated_coords_id, np.atleast_2d(drawn_vector).T), axis=1)
    # print generated_data.shape, all_data.shape
    result = np.concatenate((all_data, generated_data))
    return result

# dummy for internal use
def __predictGP(gp, train_X, train_Y, test_location):
    cholesky = gp.Cholesky(train_X)
    weights = gp.GPWeights(locations=train_X, current_location=test_location, cholesky=cholesky)
    variance = gp.GPVariance(locations=train_X, current_location=test_location, cholesky=cholesky)
    mean = gp.GPMean(measurements=train_Y, weights=weights)
    return mean, variance


if __name__ == "__main__":
    seed = 10
    selected_slots = [2,16]
    selected_slots = [16]
    for slot in selected_slots:
        # GenerateFinalDataset(slot)
        GenerateRandomRobotDataset(slot_number=slot, seed=seed)
        pass
    """
    slot_number = 16
    f = np.genfromtxt( '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/noise_generated_values_slot_' + str(
        slot_number) + '.txt')
    print f[:, :-1]
    np.savetxt(fname='../datasets/robot/selected_slots/slot_' + str(slot_number) + '/generated_id_coords_slot_' + str(
        slot_number) + '.txt', X= f[:, :-1], fmt='%10.2f')
    """