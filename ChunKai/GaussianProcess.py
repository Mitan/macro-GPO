import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from matplotlib import pyplot as pl
from matplotlib import cm

from Vis2d import Vis2d


class GaussianProcess:
    def __init__(self, covariance_function, noise_variance=0, mean_function=0.0):
        """ @param mean_function: constant mean. TODO: Change to nonstatic mean function rather than a simple constant
        """
        self.covariance_function = covariance_function
        self.noise_variance = noise_variance
        self.mean_function = mean_function

    def CovarianceFunction(self, s1, s2):
        return self.covariance_function.Cov(s1, s2)

    def CovarianceMesh(self, col_array, row_array):
        """
        @param col_array, row - array of shape (number of dimensions * number of data points)
        @return covariance matrix between physical states presented by col and row
        """
        columns = col_array.shape[0]
        rows = row_array.shape[0]
        covMat = np.zeros((columns, rows), float)
        for y in xrange(columns):
            for x in xrange(rows):
                covMat[y, x] = self.CovarianceFunction(row_array[x, :], col_array[y, :])
        return covMat

    def GPMean(self, locations, measurements, current_location, weights=None):
        """
        Return the posterior mean for measurements while the robot is in a particular augmented state

        @param weights - row vector of weight space interpretation of GP regression
        """

        if weights == None: weights = self.GPWeights(locations, current_location)

        # Obtain mean
        mean = np.dot(weights, measurements - np.ones(measurements.shape) * self.mean_function) + self.mean_function

        return mean

    def __GPVariance(self, locations, current_location, weights=None, cov_query=None):
        """
        Return the posterior variance for measurements while the robot is in a particular augmented state
        Warning: This method of computing the posterior variance is numerically unstable.

        @param weights - row vector of weight space interpretation of GP regression
        """

        if weights == None: weights = self.GPWeights(locations, current_location)
        if cov_query == None: cov_query = self.GPCovQuery(locations, current_location)

        # Obtain predictive variance by direct multiplication of
        prior_variance = self.CovarianceFunction(np.atleast_2d(current_location), np.atleast_2d(current_location))
        variance = prior_variance - np.dot(weights, cov_query.T)  # Numerically unstable component.

        return variance + self.noise_variance


    def GPBatchVariance(self, history_current, current_prior, cholesky):
        """
         history_locations - (n,n) matix - prio
        history_current - (n,k) matrix
        current_prior - (k,k) matrix
        cholesky - Cholesky decomposition of history_locations
        @ return (k,k) covariance
        """
        # similar to Alg 2.1 of GPML book. Should be (n, k) matrix
        v = linalg.solve_triangular(cholesky, history_current, lower = True)
        #should be (k,k) matrix
        return current_prior + np.dot(v.T, v)

    def GPVariance2(self, locations, current_location, cholesky=None, cov_query=None):
        """
        Return the posterior variance for measurements while the robot is in a particular augmented state

        @param cholesky - lower triangular matrix of chol decomposition of covariance matrix for training points
        @param cov_query - matrix of covariancs
        """
        if cholesky == None: cholesky = self.GPCholTraining(locations)
        if cov_query == None: cov_query = self.GPCovQuery(locations, current_location)

        prior_variance = self.CovarianceFunction(np.atleast_2d(current_location), np.atleast_2d(current_location))
        tv = linalg.solve_triangular(cholesky, cov_query.T, lower=True)
        variance = prior_variance - np.dot(tv.T, tv)

        return variance + self.noise_variance

    def GPWeights(self, locations, current_location, cholesky=None, cov_query=None):
        """
        Get a row vector of weights assuming a weight space view

        @param cholesky - lower triangular matrix of chol decomposition of covariance matrix for training points
        @param cov_query - matrix of covariancs
        """

        if cholesky == None: cholesky = self.GPCholTraining(locations)
        if cov_query == None: cov_query = self.GPCovQuery(locations, current_location)

        # Weights by matrix division using cholesky decomposition
        weights = linalg.cho_solve((cholesky, True), cov_query.T).T

        return weights


    def GPBatchWeights(self, history_current, cholesky):
        """
        history_current - (n,k) matrix - covariances between history values and new points
        current_prior - (k,k) matrix
        cholesky - Cholesky decomposition of history_locations
        @ return (k, n) matrix
        """
        # similar to Alg 2.1 of GPML book. Should be (n, k) matrix
        # todo avoid computation of v twice
        v = linalg.solve_triangular(cholesky, history_current, lower = True)
        weights_transposed = linalg.solve_triangular(cholesky.T, v, lower = False)
        return weights_transposed.T

    def GPCholTraining(self, locations):
        # Covariance matrix between existing data points
        cov_data = self.CovarianceMesh(locations, locations)

        # Cholesky decomposition for numerically stable inversion
        cholesky = np.linalg.cholesky(cov_data + self.noise_variance * np.identity(cov_data.shape[0]))

        return cholesky

    def GPCovQuery(self, locations, current_location):
        """
        Return matrix of covariances between test point and training points
        """
        # Covariance of query point to data points (row vector)
        cov_query = self.CovarianceMesh(np.atleast_2d(current_location), locations)

        return cov_query

    def GetWeightsAndVariance(self, history_locations, current_physical_state):
        # assume that both inputs are np 2D-arrays
        n = history_locations.shape[0]
        k = current_physical_state.shape[0]
        # assume that both inputs are 2-d np arrays
        cholesky = self.GPCholTraining(history_locations)
        # Sigma_n_k
        # n rows, k columns
        new_points_covariance = self.CovarianceMesh(current_physical_state, history_locations)
        weights_transposed = np.zeros((n,k ))
        # now we need to solve a linear system for every new point
        for i in range(k):
            current_covariances = new_points_covariance[:, i]
            current_solutions = linalg.cho_solve((cholesky, True), current_covariances)
            weights_transposed[:, i] = current_solutions
        # should be (k, n)
        weights = weights_transposed.T
        variance = self.CovarianceMesh(current_physical_state, current_physical_state) - np.dot(weights, new_points_covariance)
        return weights, variance





    def GPGenerate(self, predict_range=((0, 1), (0, 1)), num_samples=(20, 20), seed=142857):
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
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        assert drawn_vector.shape[0] == npoints

        return MapValueDict(points, drawn_vector)

    ### Test Suites: Inefficient Visualizers
    ###
    def GPVisualize1D(self, locations, measurements, predict_range=(0, 1), num_samples=1000):
        """
        Visualize posterior in graphical form
        NOTE: very ineffecient since we are using the weight space view to vizualize this
        """

        # Grid points
        x = np.atleast_2d(np.linspace(predict_range[0], predict_range[1], num_samples, endpoint=False)).T

        # Compute predictions - very inefficient because we are using the weight space view
        predicted_mean = [0.0] * num_samples
        predicted_variance = [0.0] * num_samples
        for i in xrange(num_samples):
            predicted_mean[i] = self.GPMean(locations, measurements, x[i])[0]
            predicted_variance[i] = self.GPVariance2(locations, x[i])[0]

        # Plot posterior mean and variances
        pl.plot(x, self.GPRegressionTestEnvironment(x), 'r:', label=u'$f(x)$')
        pl.plot(locations, measurements, 'r.', markersize=10, label=u'Observations')
        pl.plot(x, predicted_mean, 'b-', label=u'Prediction')
        pl.fill(np.concatenate([x, x[::-1]]),
                np.concatenate([predicted_mean - 1.9600 * np.sqrt(predicted_variance),
                                (predicted_mean + 1.9600 * np.sqrt(predicted_variance))[::-1]]),
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
        pl.xlabel('$x$')
        pl.ylabel('$f(x)$')
        pl.legend(loc='upper left')

        pl.show()

    def GPVisualize2D(self, locations, measurements, predict_range=((0, 1), (0, 1)), num_samples=(100, 100)):
        """
        """

        grid_res = [float(predict_range[x][1] - predict_range[x][0]) / float(num_samples[x]) for x in xrange(2)]

        # Meshed grid points
        col = np.arange(predict_range[0][0], predict_range[0][1], grid_res[0])
        row = np.arange(predict_range[1][0], predict_range[1][1], grid_res[1])
        gridc, gridr = np.meshgrid(row, col)

        # Compute predictions
        predicted_mean = np.zeros(gridc.shape)
        predicted_variance = np.zeros(gridc.shape)

        for c in xrange(col.size):
            for r in xrange(row.size):
                predicted_mean[c, r] = self.GPMean(locations, measurements, np.array([col[c], row[r]]))[0]
                predicted_variance[c, r] = self.GPVariance2(locations, np.array([col[c], row[r]]))[0]

        # Plot posterior means and variances
        fig = pl.figure(figsize=pl.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax.plot_surface(gridr, gridc, predicted_mean, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.scatter(locations[:, 0], locations[:, 1], measurements, c='r', marker='o')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        fig.colorbar(surf, shrink=0.5, aspect=10)

        pl.show()

    def GPRegressionTest(self, test="1d"):
        """
        Test GPR and displays results on screen
        """

        if test == "1d":
            # Generate history
            locations = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
            measurements = self.GPRegressionTestEnvironment(locations, "1d")

            self.GPVisualize1D(locations, measurements, (0, 10))

        elif test == "2dgaussian":
            # Generate history
            locations = np.atleast_2d([[0, 0], [0, 1], [1, 0]])
            measurements = self.GPRegressionTestEnvironment(locations, "2dgaussian")

            self.GPVisualize2D(locations, measurements, ((-10, 10), (-10, 10)), (50, 50))

        elif test == "2dmix2gaussian":
            # Generate history

            mesh = np.mgrid[-10:10.01:4, -10:10.01:4]
            mesh = mesh.reshape(2, -1).T

            locations = np.atleast_2d(mesh)
            measurements = self.GPRegressionTestEnvironment(locations, "2dmix2gaussian")

            self.GPVisualize2D(locations, measurements, ((-10, 10), (-10, 10)), (20, 20))

        elif test == "2dmixed":

            mesh = np.mgrid[-10:10.01:4, -10:10.01:4]
            mesh = mesh.reshape(2, -1).T

            locations = np.atleast_2d(mesh)
            measurements = self.GPRegressionTestEnvironment(locations, "2dmixed")

            self.GPVisualize2D(locations, measurements, ((-10, 10), (-10, 10)), (20, 20))

    def GPRegressionTestEnvironment(self, loc, test="1d"):
        if test == "1d":
            # Environment field of xsin(x)
            return loc * np.sin(loc)
        elif test == "2dgaussian":
            # 2d multivariate distribution centered at (0)
            var = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
            return np.apply_along_axis(lambda xy: var.pdf(xy), 1, loc)
        elif test == "2dmix2gaussian":
            var1 = multivariate_normal(mean=[2, 5], cov=[[4, 0], [0, 1]])
            var2 = multivariate_normal(mean=[-3, -5], cov=[[1, 0], [0, 1]])

            return np.apply_along_axis(lambda xy: var1.pdf(xy) + var2.pdf(xy), 1, loc)

        elif test == "2dmixed":
            var1 = multivariate_normal(mean=[-3, 3], cov=[[4, 0], [0, 4]])
            var2 = lambda xy: 0.01 * xy[0] * np.sin(0.05 * xy[0])

            return np.apply_along_axis(lambda xy: var1.pdf(xy) + var2(xy), 1, loc)

    def GPGenerateTest(self, predict_range=((-1, 1),), num_samples=(30,)):
        assert (len(predict_range) == len(num_samples))
        ndims = len(predict_range)

        mapping = self.GPGenerate(predict_range, num_samples)

        if ndims > 2:
            print "Dimensions > 2. Unable to display function"
            return

        if ndims == 1:

            # Grid points
            x = np.atleast_2d(np.linspace(predict_range[0][0], predict_range[0][1], num_samples[0], endpoint=False)).T
            mapping_v = np.vectorize(mapping)
            y = mapping_v(x)

            # Plot posterior mean and variances
            pl.plot(x, y, 'r:', label=u'$f(x)$')
            pl.xlabel('$x$')
            pl.ylabel('$f(x)$')
            pl.legend(loc='upper left')

            pl.show()
        else:

            grid_res = [float(predict_range[x][1] - predict_range[x][0]) / float(num_samples[x]) for x in xrange(2)]
            # Meshed grid points
            col = np.arange(predict_range[0][0], predict_range[0][1], grid_res[0])
            row = np.arange(predict_range[1][0], predict_range[1][1], grid_res[1])

            ground_truth = np.zeros(num_samples)
            for a in xrange(num_samples[0]):
                for b in xrange(num_samples[1]):
                    ground_truth[a][b] = mapping((col[a], row[b]))

            vis2d = Vis2d()
            vis2d.MapPlot(predict_range[0] + predict_range[1], ground_truth=ground_truth)


"""
=== Covariance Functions ===

Defines the common covariance functions

"""


class CovarianceFunction:
    """
    Just a dummy class to invoke more structure
    """

    def __init__(self):
        pass


class SquareExponential(CovarianceFunction):
    def __init__(self, length_scale, signal_variance):
        """
        @param: length_scale l - array or list containing the length scales for each dimension
        @param: signal variance sigma_f_squared - float containing the signal variance
        """

        self.length_scale = np.atleast_2d(length_scale)
        self.signal_variance = signal_variance

    def Cov(self, physical_state_1, physical_state_2):
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        squared = np.dot(diff, np.divide(diff, self.length_scale).T)
        return self.signal_variance * np.exp(-0.5 * squared)


class MapValueDict():
    def __init__(self, locations, values, epsilon=None):
        """
        @param epsilon - minimum tolerance level to determine equivalence between two points
        """
        self.locations = locations
        self.values = values

        if not epsilon == None:
            self.epsilon = epsilon
            return

        ndims = locations.shape[1]
        self.epsilon = np.zeros((ndims,))
        for dim in xrange(ndims):
            temp = list(set(np.squeeze(locations[:, dim]).tolist()))
            temp = sorted(temp)
            self.epsilon[dim] = (min([temp[i] - temp[i - 1] for i in xrange(1, len(temp))])) / 4

    def __call__(self, query_location):
        """
        Search for nearest grid point iteratively. Uses L1 norm as the distance metric
        """
        bi = -1
        bd = None
        for i in xrange(self.locations.shape[0]):
            d = np.absolute(np.atleast_2d(query_location) - self.locations[i, :])
            l1 = np.sum(d)
            if np.all(d <= self.epsilon) and (bd == None or l1 < bd):
                bd = l1
                bi = i

        assert not bd == None, "No close enough match found for query location " + str(query_location)

        return self.values[bi]


if __name__ == "__main__":
    # Generation Tests
    covariance_function = SquareExponential(0.05, 1)
    gp1d = GaussianProcess(covariance_function)
    gp1d.GPGenerateTest(predict_range=((-1, 1),), num_samples=(100,))

    covariance_function = SquareExponential(30, 1)
    gp2d = GaussianProcess(covariance_function)
    gp2d.GPGenerateTest(predict_range=((0, 1), (0, 1)), num_samples=(30, 30))

    # Regression Tests
    # 1 Dimensional Test #
    covariance_function = SquareExponential(1.5, 1)
    gp1d = GaussianProcess(covariance_function)
    gp1d.GPRegressionTest("1d")  # xsinx

    gp1d2 = GaussianProcess(covariance_function, 1)  # add in some noise
    gp1d2.GPRegressionTest("1d")

    # 2 Dimensional Tests #
    covariance_function = SquareExponential(np.array([1.5, 1.5]), 1)
    gp2d = GaussianProcess(covariance_function)
    gp2d.GPRegressionTest("2dgaussian")  # single gaussian

    gp2d2 = GaussianProcess(covariance_function)
    gp2d2.GPRegressionTest("2dmix2gaussian")  # two gaussians

    gp2d3 = GaussianProcess(covariance_function)
    gp2d3.GPRegressionTest("2dmixed")  # mixture of two functions
