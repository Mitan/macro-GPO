from src.model.MapValueDictBase import MapValueDictBase
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class SimulatedMapValueDict(MapValueDictBase):

    def __init__(self):
        MapValueDictBase.__init__(self, locations=locs, values=vals)

    # for given state
    def GetSelectedMacroActions(self, current_state):
        pass

    def GPGenerate(self, predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=142857, noiseVariance=0):
        """
        Generates a draw from the gaussian process

        @param predict_range - map range for each dimension
        @param num_samples - number of samples for each dimension
        @return dict mapping locations to values
        """
        assert (len(predict_range) == len(num_samples))

        # Number of dimensions of the multivariate gaussian is equal to the number of grid points
        ndims = len(num_samples)
        grid_res = [float(predict_range[x][1] - predict_range[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
        npoints = reduce(lambda a, b: a * b, num_samples)

        # Mean function is assumed to be zero
        u = np.zeros(npoints)

        # List of points
        grid1dim = [slice(predict_range[x][0], predict_range[x][1], grid_res[x]) for x in xrange(ndims)]
        grids = np.mgrid[grid1dim]
        points = grids.reshape(ndims, -1).T

        # print points
        # raw_input()

        assert points.shape[0] == npoints

        # construct covariance matrix
        cov_mat = self.CovarianceMesh(points, points)

        # Draw vector
        np.random.seed(seed=seed)
        # these are noiseless observations
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        # add noise to them
        noise_components = np.random.normal(0, np.math.sqrt(noiseVariance), npoints)
        assert drawn_vector.shape == noise_components.shape
        assert drawn_vector.shape[0] == npoints
        drawn_vector_with_noise = np.add(drawn_vector, noise_components)

        # print points
        # print drawn_vector

        return MapValueDict(points, drawn_vector_with_noise)

    def GenerateSimulatedModel(length_scale, signal_variance, noise_variance, save_folder, seed, predict_range,
                               num_samples, mean_function):
        covariance_function = SquareExponential(length_scale, signal_variance=signal_variance,
                                                noise_variance=noise_variance)
        # Generate a drawn vector from GP with noise
        gpgen = GaussianProcess(covariance_function, mean_function=mean_function)
        m = gpgen.GPGenerate(predict_range=predict_range, num_samples=num_samples, seed=seed,
                             noiseVariance=noise_variance)
        # write the dataset to file
        m.WriteToFile(save_folder + "dataset.txt")
        return m