import math

import GPy
import numpy as np
import scipy

from src.GaussianProcess import SquareExponential, GaussianProcess


# exact values based on GP prediction
def GenerateExactValues(slot_number):
    input_data_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/slot_' + str(slot_number) + '.txt'
    hypers_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/hypers_' + str(slot_number) + '.txt'

    real_coordinates_file = '../datasets/robot/processing/coordinates.txt'
    fake_coordinates_file = '../datasets/robot/processing/fake_coordinates.txt'
    output_filename = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/noise_generated_values_slot_' + str(
        slot_number) + '.txt'

    output_file = open(output_filename, 'w')

    all_real_coords = np.genfromtxt(real_coordinates_file)
    all_fake_coords = np.genfromtxt(fake_coordinates_file)
    all_data = np.genfromtxt(input_data_file)
    hypers = np.genfromtxt(hypers_file)

    id_list = all_data[:, 0]
    missing_indexes = __FindMissingIndexes(id_list)

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

    # Check the prediction with GPy
    """
    kernel = GPy.kern.RBF(input_dim=point_dimension, variance=signal_variance, lengthscale=length_scale, ARD=True)
    Y_upd = Y - mean
    m = GPy.models.GPRegression(X, Y_upd, kernel, noise_var=noise_variance, normalizer=False)
    """

    for missing_index in missing_indexes:
        real_coord_line = all_real_coords[missing_index - 1, :]
        assert real_coord_line[0] == missing_index
        test_location = real_coord_line[1:]
        mu = __predictGP(gp=gp, train_X=X, train_Y=Y, test_location=test_location)
        # noise
        mu = mu + np.random.normal(loc=0, scale = math.sqrt(noise_variance))
        mu = round(mu, 4)
        """
        mu_gpy, _ = m.predict(np.atleast_2d(test_location), full_cov=True)
        print mu - (mu_gpy+ mean)[0,0]
        """
        output_file.write(
            str(missing_index) + ' ' + str(test_location[0]) + ' ' + str(test_location[1]) + ' ' + str(mu) + '\n')

    for fake_coord_line in all_fake_coords:
        test_location = fake_coord_line[1:3]
        mu = __predictGP(gp=gp, train_X=X, train_Y=Y, test_location=test_location)
        # noise
        mu = mu + np.random.normal(loc=0, scale=math.sqrt(noise_variance))
        mu = round(mu, 4)
        output_file.write(
            str(fake_coord_line[0]) + ' ' + str(test_location[0]) + ' ' + str(test_location[1]) + ' ' + str(mu) + '\n')

    output_file.close()

# dummy for internal use
def __predictGP(gp, train_X, train_Y, test_location):
    cholesky = gp.Cholesky(train_X)
    weights = gp._gp_weights(locations=train_X, current_location=test_location, cholesky=cholesky)
    return gp.GPMean(measurements=train_Y, weights=weights)[0, 0]


# find point in the dataset which do not have a value
def __FindMissingIndexes(existing_indexes):
    # id of all points
    allowed_range = range(1, 55)
    return [i for i in allowed_range if not i in existing_indexes]


def GenerateFinalDataset(slot_number):
    GenerateExactValues(slot_number)
    real_values_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/slot_' + \
                       str(slot_number) + '.txt'
    generated_values_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + \
                            '/noise_generated_values_slot_' + str(slot_number) + '.txt'
    output_file = '../datasets/robot/selected_slots/slot_' + str(slot_number) + '/noise_final_slot_' + \
                       str(slot_number) + '.txt'
    filenames = [real_values_file, generated_values_file]
    with open(output_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

if __name__ == "__main__":
    selected_slots = [2,16]
    selected_slots = [16]
    for slot in selected_slots:
        GenerateFinalDataset(slot)
