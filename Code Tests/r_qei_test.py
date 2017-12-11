import random

import GPy
import numpy as np
import rpy2.robjects as robjects  # This initializes R
from src.r_qei import newQEI


class GPyPredictor:
    def __init__(self, length_scale, signal_variance, noise_variance, locations, Y):
        point_dimension = locations.shape[1]
        kernel = GPy.kern.RBF(input_dim=point_dimension, variance=signal_variance, lengthscale=length_scale, ARD=True)
        self.m = GPy.models.GPRegression(locations, Y, kernel, noise_var=noise_variance, normalizer=False)

    def predict(self, new_location):
        return self.m.predict(new_location, full_cov=True)


if __name__ == "__main__":
    for k in range(100):
        # init dimensions
        point_dimension = random.randint(2, 9)
        num_points = 20

        # init hypers
        noise_variance = random.random()  # Random float x, 0.0 <= x < 1.0
        signal_variance = random.uniform(2, 5)
        length_scale = [random.uniform(5., 10.0) for i in range(point_dimension)]

        # init setting
        locations = np.random.uniform(-10., 15., (num_points, point_dimension))

        Y = np.zeros((num_points, 1))
        for i in range(num_points):
            current_point = locations[i, :]
            Y[i, 0] = -0.2 * current_point[0] + current_point[0] ** 2 + np.sum(current_point)

        batch_size = random.randint(2, 9)
        new_location = np.random.uniform(-3., 3., (batch_size, point_dimension))

        qei = newQEI(length_scale=length_scale, signal_variance=signal_variance, noise_variance=noise_variance,
                     locations=locations, Y=Y)

        qei_value = qei.acquisition(new_location)

        mu_R = robjects.r['model_mean'](new_location, qei.r_model)
        var_R = robjects.r['model_var'](new_location, qei.r_model)
        cov_R = np.array(robjects.r['model_covariance'](new_location, qei.r_model))

        gpy = GPyPredictor(length_scale=length_scale, signal_variance=signal_variance, noise_variance=noise_variance,
                           locations=locations, Y=Y)

        mu_gpy, var_gpy = gpy.predict(new_location)

        # print cov_R - var_gpy, noise_variance

        eps_tolerance = 10 ** -5

        # compare
        """
        assert mu_R.shape == mu_gpy.shape
        assert np.linalg.norm(mu_R - mu_gpy) < eps_tolerance, "difference is %r" % np.linalg.norm(mu_R - mu_gpy)
        print cov_R, var_gpy

        assert var_R.shape == var_gpy.shape
        assert np.linalg.norm(var_R - var_gpy) < eps_tolerance, "difference is %r" % np.linalg.norm(var - var_gpy)
        """
        # print('error mean:', np.linalg.norm(mu_gpy[:, 0] - np.asarray(mu_R)))
        # print('error var:', np.linalg.norm(var_gpy[:, 0] - np.asarray(var_R)) /np.linalg.norm(var_gpy[:, 0]) * 100, '%')
        # print('error var:', var_gpy[:, 0] - np.asarray(var_R), noise_variance)

        # todo nb
        """
        Gpy calculates covariance matrix with noise variance,
        qEI calculates it without  noise variance - doesn't add it

        """

    '''
    # Compare the multipoint expected improvement given by DiceOptim
    # against sampling. Warning: sometimes qEI is arbitrarily wrong see
    # qEI_problem.R
    mean, cov = self.predict_f_full_cov(X)
    fmin = min(self.Y.value)
    N = 100000
    imp = np.zeros(N)
    for k in range(N):
        Y = np.random.multivariate_normal(mean[:, 0], cov[:, :, 0])[:, None]
        imp[k] = min(fmin, min(Y)) - fmin

    print('EI via sampling:', np.mean(imp))
    print('EI via DiceOptim:', opt_val)
    '''
