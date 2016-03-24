from datetime import datetime
import os
import math
from SimulatedDataSetsHypers import __Ackley

from SampleFunctionBuilder import GetSampleFunction
from TreePlan import *
from GaussianProcess import GaussianProcess
from GaussianProcess import SquareExponential
from Vis2d import Vis2d


class TreePlanTester:
    def __init__(self, simulate_noise_in_trials=True):
        """
        @param simulate_noise_in_trials: True if we want to add in noise artificially into measurements
        False if noise is already presumed to be present in the data model
        """
        self.simulate_noise_in_trials = simulate_noise_in_trials
        # if reward_model == "Linear":
        # for batch case z is a list of k measurements
        self.reward_function = lambda z: sum(z)
        """
        elif reward_model == "Positive_log":
            self.reward_function = lambda z: math.log(z) if z > 1 else 0.0
        elif reward_model == "Step1mean":  # Step function with mean 0
            self.reward_function = lambda z: 1.0 if z > 1 else 0.0
        elif reward_model == "Step15mean":
            self.reward_function = lambda z: 1.0 if z > 1.5 else 0.0
        else:
            assert False, "Unknown reward type"

        self.bad_places = bad_places
        self.sd_bonus = sd_bonus
        """

    def InitGP(self, length_scale, signal_variance, noise_variance, mean_function=0.0):
        """
        @param length_scale: list/nparray containing length scales of each axis respectively
        @param signal_variance
        @param noise_variance
        @param mean_function

        Example usage: InitGP([1.5, 1.5], 1, 0.1)
        """
        self.covariance_function = SquareExponential(np.array(length_scale), signal_variance)
        self.gp = GaussianProcess(self.covariance_function, noise_variance, mean_function=mean_function)
        self.noise_variance = noise_variance

    def InitEnvironment(self, environment_noise, model):
        """
        @param environment noise - float for variance of zero mean gaussian noise present in the actual environment
        @param model - function taking in a numpy array of appropriate dimension and returns the actual (deterministic) reading

        Example usage: InitEnvironment(0.1, lambda xy: multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]]).pdf(xy))
        Makes the environment with 0.1 noise variance with a mean following that of a standard multivariate normal
        """
        self.environment_noise = environment_noise
        self.model = model

    def InitPlanner(self, grid_domain, grid_gap):
        """
        Creates a planner. For now, we only allow gridded/latticed domains.

        @param grid_domain - 2-tuple of 2-tuples; Each tuple for a dimension, each containing (min, max) for that dimension
        @param grid_gap - float, resolution of our domain
        @param epsilon - maximum allowed policy loss
        @param gamma - discount factor
        @param H - int, horizon
        @param batch_size - int, number of points in a batch (k)

        Example usage:
        InitPlanner(((-10, 10), (-10, 10)), 0.2, 100.0, 1.0, 5)

        for the following desired parameters:
        ----------------------------------------
        grid_domain = ((-10, 10), (-10, 10))
        grid_gap = 0.2
        epsilon = 100.0
        gamma = 1.0
        H = 5
        ----------------------------------------
        """

        self.grid_domain = grid_domain
        self.grid_gap = grid_gap

    def InitTestParameters(self, initial_physical_state, past_locations):
        """
        Defines the initial state and other testing parameters.
        Use only after the environment model has been defined
        Initial measurements are drawn from the previously defined environment model
        Though not an enforced requirement, it is generally sensible to have the initial_physical_state as one of the past locations

        @param initial_physical_state - numpy array of appropriate dimension
        @param past_locations - numpy array of shape |D| * |N| where D is the number of locations already visited and N is the number of dimensions

        Example Usage:
        initial_physical_state = np.array([1.0, 1.0])
        past_locations = np.array([[-1.0, -1.0], [1.0, 1.0]])

        InitTestParameters(initial_physical_state, past_locations)
        """

        self.initial_physical_state = initial_physical_state
        self.past_locations = past_locations

        # Compute measurements
        self.past_measurements = None if self.past_locations is None else np.apply_along_axis(self.model, 1,
                                                                                              past_locations)

    def Test(self, num_timesteps_test, H, batch_size, alg_type, my_nodes_func, beta, debug=False, save_per_step=True,
             save_folder="default_results/"):
        """ Pipeline for testing
        @param num_timesteps_test - int, number of timesteps we should RUN the algo for. Do not confuse with search horizon
        """

        # s_0, d_0
        x_0 = AugmentedState(self.initial_physical_state,
                             initial_history=History(self.past_locations, self.past_measurements))
        state_history = [x_0]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        total_reward = 0
        measurement_history = []
        reward_history = []
        nodes_expanded_history = []
        base_measurement_history = []
        for time in xrange(num_timesteps_test):
            tp = TreePlan(self.grid_domain, self.grid_gap, self.gp,
                          batch_size=batch_size, number_of_nodes_function=my_nodes_func, horizon=H, beta=beta)

            if alg_type == 'qEI':
                _, a, _ = tp.qEI(x_0)
            elif alg_type == 'UCB':
                v, a, nodes_expanded = tp.StochasticFull(x_0, 1)
            elif alg_type == 'Non-myopic':
                _, a, _ = tp.StochasticFull(x_0, H)
            elif alg_type == 'MLE':
                _, a, _ = tp.MLE(x_0, H)
            else:
                raise ValueError("wrong type")
            # _, a, nodes_expanded = tp.StochasticFull(x_0, self.H)

            x_temp = tp.TransitionP(x_0, a)
            # Draw an actual observation from the underlying environment field and add it to the our measurements
            print x_temp.physical_state[0].shape
            # for batch case x_temp.physical_states is a list
            # single_agent_state is a position of one agent in a batch
            baseline_measurements = [self.model(single_agent_state) for single_agent_state in x_temp.physical_state]
            # noise components is a set of noises for k agents
            if self.simulate_noise_in_trials:
                noise_components = np.random.normal(0, math.sqrt(self.noise_variance), batch_size)
            else:
                noise_components = [0 for i in range(batch_size)]
            percieved_measurements = baseline_measurements

            x_next = tp.TransitionH(x_temp, percieved_measurements)

            # Update future state
            x_0 = x_next

            reward_obtained = self.reward_function(percieved_measurements)

            # Accumulated measurements
            reward_history.append(reward_obtained)
            total_reward += reward_obtained
            measurement_history.append(percieved_measurements)
            base_measurement_history.append(baseline_measurements)
            # total_nodes_expanded += nodes_expanded
            # nodes_expanded_history.append(nodes_expanded)

            if debug:
                print "A = ", a
                print "M = ", percieved_measurements
                print "X = "
                print "Noise = ", noise_components
                print x_0.to_str()

            # Add to plot history
            state_history.append(x_0)

            if save_per_step:
                self.Visualize(state_history=state_history, display=False,
                               save_path=save_folder + "step" + str(time))
                # Save to file
                f = open(save_folder + "step" + str(time) + ".txt", "w")
                f.write(x_0.to_str() + "\n")
                f.write("Total accumulated reward = " + str(total_reward))
                f.close()

        # Save for the whole trial
        self.Visualize(state_history=state_history, display=False, save_path=save_folder + "summary")
        # Save to file
        f = open(save_folder + "summary" + ".txt", "w")
        f.write(x_0.to_str() + "\n")
        f.write("===============================================")
        f.write("Measurements Collected\n")
        f.write(str(measurement_history) + "\n")
        f.write("Base measurements collected\n")
        f.write(str(base_measurement_history) + "\n")
        f.write("Total accumulated reward = " + str(total_reward) + "\n")
        f.write("Nodes Expanded per stage\n")
        # f.write(str(nodes_expanded_history) + "\n")
        # f.write("Total nodes expanded = " + str(total_nodes_expanded))
        f.close()

        return state_history, reward_history, nodes_expanded_history, base_measurement_history

    def Visualize(self, state_history, display=True, save_path=None):
        """ Visualize 2d environments
        """

        XGrid = np.arange(self.grid_domain[0][0], self.grid_domain[0][1] - 1e-10, self.grid_gap)
        YGrid = np.arange(self.grid_domain[1][0], self.grid_domain[1][1] - 1e-10, self.grid_gap)
        XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        ground_truth = np.vectorize(lambda x, y: self.model([x, y]))

        # Plot graph of locations
        vis = Vis2d()
        vis.MapPlot(grid_extent=[self.grid_domain[0][0], self.grid_domain[0][1], self.grid_domain[1][0],
                                 self.grid_domain[1][1]],
                    ground_truth=ground_truth(XGrid, YGrid),
                    path_points=[x.physical_state for x in state_history],
                    display=display,
                    save_path=save_path)


def Random(initial_state, horizon, batch_size, alg_type, my_func, beta, grid_gap_=0.5, length_scale=(0.1, 0.1),
           num_timesteps_test=20,
           noise_variance=10 ** -5,

           seed=142857, save_folder=None, save_per_step=False,
           ):
    """
    Assume a map size of [0, 1] for both axes
    """
    covariance_function = SquareExponential(length_scale, 1)
    gpgen = GaussianProcess(covariance_function)
    m = gpgen.GPGenerate(predict_range=((-15, 15), (-15, 15)), num_samples=(20, 20), seed=seed)
    m = __Ackley
    TPT = TreePlanTester(simulate_noise_in_trials=True)

    TPT.InitGP(length_scale=(1.76030067316, 10.7911469234), signal_variance=0.571823797414, noise_variance=0.0430165337672, mean_function= 1.20093709782)
    TPT.InitEnvironment(environment_noise=noise_variance, model=m)

    TPT.InitPlanner(grid_domain=((-15, 15), (-15, 15)), grid_gap=grid_gap_)

    TPT.InitTestParameters(initial_physical_state=initial_state,
                           past_locations=initial_state)
    TPT.Test(num_timesteps_test=num_timesteps_test, H=horizon, batch_size=batch_size, alg_type=alg_type,
             my_nodes_func=my_func, beta=beta, debug=False, save_folder=save_folder,
             save_per_step=save_per_step)


def initial_state(batch_size):
    if batch_size == 2:
        return np.array([[4.0, 4.0], [-3.0, -3.0]])
    elif batch_size == 3:
        return np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]])
    elif batch_size == 4:
        return np.array([[0.2, 0.2], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]])
    else:
        raise Exception("wrong batch size")


if __name__ == "__main__":
    # assert len(sys.argv) == 2, "Wrong number of arguments"

    # initial_state = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]])
    # initial_state = np.array([[0.2, 0.2], [0.8, 0.8]])
    save_trunk = "./tests/"
    # my_batch_size = 2


    beta = 0.0
    horizons = [3]
    for h in horizons:
        for b in range(2, 3):
            print b, h
            print datetime.now()
            for i in xrange(111, 112):
                f = lambda t: GetSampleFunction(h, t)
                my_save_folder = save_trunk + "seed" + str(i) + "_b" + str(b) + "_h" + str(h)
                my_initial_state = initial_state(b)

                """
                Random(my_initial_state, h, b, 'Non-myopic', f, beta, length_scale=(0.1, 0.1), seed=i,
                       save_folder=my_save_folder + '_non-myopic' + "/",
                       save_per_step=True, num_timesteps_test=7)
                """
                Random(my_initial_state, h, b, 'UCB', f, beta, length_scale=(0.1, 0.1), seed=i,
                       save_folder=my_save_folder + '_ucb' + "/",
                       save_per_step=True, num_timesteps_test=12)
                """
                Random(my_initial_state, h, b, 'Non-myopic', f, beta, length_scale=(0.1, 0.1), seed=i,
                       save_folder=my_save_folder + '_non-myopic' + "/",
                       save_per_step=True, num_timesteps_test=12)
                """
            print datetime.now()
            print
            # Transect(seed=i)

            # print "Performing sanity checks"
            # SanityCheck()
            # print "Performing Exploratory"
            # Exploratory(1.0) # This goes to weird places
            # print "Performing Exploratory 2"
            # Exploratory(0.5)
