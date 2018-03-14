from src.GaussianProcess import GaussianProcess, SquareExponential
from src.enum.DatasetEnum import DatasetEnum
from src.model.MapValueDictBase import MapValueDictBase
import numpy as np

from scipy.stats import multivariate_normal


class SimulatedMapValueDict(MapValueDictBase):

    def __init__(self, hyper_storer, domain_descriptor, seed=None, filename=None):
        self.dataset_type = DatasetEnum.Simulated
        self.hyper_storer = hyper_storer
        self.domain_descriptor = domain_descriptor
        if filename:
            data = np.genfromtxt(filename)
            locs = data[:, :-1]
            vals = data[:, -1]
        else:
            covariance_function = SquareExponential(length_scale=hyper_storer.length_scale,
                                                    signal_variance=hyper_storer.signal_variance,
                                                    noise_variance=hyper_storer.noise_variance)

            gp = GaussianProcess(covariance_function=covariance_function,
                                 mean_function=hyper_storer.mean_function)
            locs, vals = self.__generate_values(gp=gp,
                                                grid_domain=domain_descriptor.grid_domain,
                                                num_samples=domain_descriptor.num_samples_grid,
                                                seed=seed,
                                                noise_variance=hyper_storer.noise_variance)

        MapValueDictBase.__init__(self, locations=locs, values=vals)

        batch_size = 4
        self.macroaction_set = self.__GenerateSimpleMacroactions(batch_size)

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

    """
    def GenerateSimulatedModel(length_scale, signal_variance, noise_variance, save_folder, seed, predict_range,
                               num_samples):
        covariance_function = SquareExponential(length_scale, signal_variance=signal_variance,
                                                noise_variance=noise_variance)
        # Generate a drawn vector from GP with noise
        gpgen = GaussianProcess(covariance_function)
        m = gpgen.GPGenerate(predict_range=predict_range, num_samples=num_samples, seed=seed,
                             noiseVariance=noise_variance)
        # write the dataset to file
        m.WriteToFile(save_folder + "dataset.txt")
        return m
    """

    def PhysicalTransition(self, current_location, macroaction):

        # current_location = physical_state[-1, :]
        batch_size = macroaction.shape[0]

        repeated_location = np.asarray([current_location for i in range(batch_size)])
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

    def __GetStraightLineMacroAction(self, direction, length):
        return np.asarray([[direction[0] * i, direction[1] * i] for i in range(1, length + 1)])

    # Generates simple macroactions allowing to move straight in specified directions
    def __GenerateSimpleMacroactions(self, batch_size):
        grid_gap = self.domain_descriptor.grid_gap
        action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0), (-grid_gap, 0))
        return [self.__GetStraightLineMacroAction(direction, batch_size) for direction in action_set]


"""
def TestScenario(my_save_folder_root, h_max, seed, time_steps, num_samples, batch_size, filename=None):
    result_graphs = []

    # eps = 10 ** 10
    save_folder = my_save_folder_root + "seed" + str(seed) + "/"

    try:
        os.makedirs(save_folder)
    except OSError:
        if not os.path.isdir(save_folder):
            raise

    # this model is for observed values
    length_scale = (0.25, 0.25)
    signal_variance = 1.0
    noise_variance = 0.00001
    predict_range = ((-0.25, 2.25), (-0.25, 2.25))
    num_samples_grid = (50, 50)

    # file for storing reward histories
    # so that later we can plot only some of them

    output_rewards = open(save_folder + "reward_histories.txt", 'w')

    if filename is not None:
        m = GenerateModelFromFile(filename)
    else:
        m = GenerateSimulatedModel(length_scale=np.array(length_scale), signal_variance=signal_variance,
                                   seed=seed, noise_variance=noise_variance, save_folder=save_folder,
                                   predict_range=predict_range, num_samples=num_samples_grid)
"""
