import numpy as np
from scipy.stats import multivariate_normal

from src.dataset_model.MapValueDictBase import MapValueDictBase
from src.enum.DatasetEnum import DatasetEnum
from random import choice


class BraninMapValueDict(MapValueDictBase):

    def __init__(self, hyper_storer, domain_descriptor, batch_size , filename):
        self.dataset_type = DatasetEnum.Branin
        self.hyper_storer = hyper_storer
        self.domain_descriptor = domain_descriptor
        self.batch_size = batch_size

        data = np.genfromtxt(filename)
        locs = data[:, :-1]
        vals = data[:, -1]

        MapValueDictBase.__init__(self, locations=locs, values=vals)

        self.macroaction_set = self.__GenerateSimpleMacroactions()

        self.max_value = max(self.values)
        self.empirical_mean = np.mean(self.values)

    # for given state
    def GetSelectedMacroActions(self, current_state):
        all_available_macroactions = [self.PhysicalTransition(current_state, a)
                                      for a in self.macroaction_set]
        return [physical_state for physical_state in all_available_macroactions
                if self.__isValidMacroAction(physical_state)]

    # generates values with zero mean
    def __generate_values(self, gp, grid_domain, num_samples, seed, noise_variance):

        assert (len(grid_domain) == len(num_samples))

        # Number of dimensions of the multivariate gaussian is equal to the number of grid points
        ndims = len(num_samples)
        grid_res = [float(grid_domain[x][1] - grid_domain[x][0]) / float(num_samples[x]) for x in xrange(ndims)]
        npoints = reduce(lambda a, b: a * b, num_samples)

        # Mean function is assumed to be zero
        u = np.zeros(npoints)

        # List of points
        grid1dim = [slice(grid_domain[x][0], grid_domain[x][1], grid_res[x]) for x in xrange(ndims)]
        grids = np.mgrid[grid1dim]
        points = grids.reshape(ndims, -1).T

        # print points
        # raw_input()

        assert points.shape[0] == npoints

        # construct covariance matrix
        cov_mat = gp.CovarianceMesh(points, points)

        # Draw vector
        np.random.seed(seed=seed)
        # these are noiseless observations
        drawn_vector = multivariate_normal.rvs(mean=u, cov=cov_mat)
        # add noise to them
        noise_components = np.random.normal(0, np.math.sqrt(noise_variance), npoints)
        assert drawn_vector.shape == noise_components.shape
        assert drawn_vector.shape[0] == npoints
        drawn_vector_with_noise = np.add(drawn_vector, noise_components)

        return points, drawn_vector_with_noise


    def PhysicalTransition(self, current_location, macroaction):

        # current_location = physical_state[-1, :]

        repeated_location = np.asarray([current_location for i in range(self.batch_size)])
        # repeated_location = np.tile(current_location, batch_size)

        assert repeated_location.shape == macroaction.shape
        # new physical state is a batch starting from the current location (the last element of batch)
        new_physical_state = np.add(repeated_location, macroaction)

        # check that it is 2d
        assert new_physical_state.ndim == 2
        return new_physical_state

    def __isValidMacroAction(self, physical_state):
        # TODO: ensure scalability to multiple dimensions
        # TODO: ensure epsilon comparison for floating point comparisons (currently comparing directly like a noob)

        # Physical state is a macro-action (batch)

        # both should be equal to 2, since the points are 2-d.
        # the first dimension is the length of state. should be equal to batch size
        #  but can't compare because of the first step

        #  print physical_state, a
        # assert physical_state.shape[1] == a.shape[1]
        # new_state = self.PhysicalTransition(physical_state, a)
        # assert new_state.shape == a.shape
        # print new_state
        assert physical_state.shape[1] == 2
        ndims = 2
        eps = 0.001
        # a.shape[0] is batch_size
        for i in range(physical_state.shape[0]):
            current_agent_postion = physical_state[i, :]
            for dim in xrange(ndims):
                if current_agent_postion[dim] < self.domain_descriptor.grid_domain[dim][0] or current_agent_postion[
                    dim] >= \
                        self.domain_descriptor.grid_domain[dim][1]:
                    return False
        return True

    def __GetStraightLineMacroAction(self, direction):
        return np.asarray([[direction[0] * i, direction[1] * i] for i in range(1, self.batch_size + 1)])

    # Generates simple macroactions allowing to move straight in specified directions
    def __GenerateSimpleMacroactions(self):
        grid_gap = self.domain_descriptor.grid_gap
        action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0), (-grid_gap, 0))
        return [self.__GetStraightLineMacroAction(direction) for direction in action_set]

    def GenerateStartLocation(self):
        self.start_location = np.array([choice(list(self.locations))])

    def LoadStartLocation(self, location_filename):

        # should contain only one line
        string_locations = open(location_filename).readline().split()
        location = map(float, string_locations)
        assert len(location) == 2
        self.start_location = np.array([location])

    def get_max(self):
        return self.max_value

    def get_empirical_mean(self):
        return self.empirical_mean


