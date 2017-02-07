import os

from GaussianProcess import MapValueDict
from TreePlan import *
from Vis2d import Vis2d
from MethodEnum import Methods
from DynamicHorizon import DynamicHorizon
from HypersStorer import *


class TreePlanTester:
    def __init__(self, simulate_noise_in_trials=True, beta=0.0, bad_places=None):
        """
        @param simulate_noise_in_trials: True if we want to add in noise artificially into measurements
        False if noise is already presumed to be present in the data model
        """
        self.simulate_noise_in_trials = simulate_noise_in_trials

        self.reward_function = lambda z: sum(z)
        """
        if reward_model == "Linear":
            self.reward_function = lambda z: z
        elif reward_model == "Positive_log":
            self.reward_function = lambda z: math.log(z) if z > 1 else 0.0
        elif reward_model == "Step1mean":  # Step function with mean 0
            self.reward_function = lambda z: 1.0 if z > 1 else 0.0
        elif reward_model == "Step15mean":
            self.reward_function = lambda z: 1.0 if z > 1.5 else 0.0
        else:
            assert False, "Unknown reward type"
        """
        self.bad_places = bad_places
        self.beta = beta

    # just sets the parameters
    def InitGP(self, length_scale, signal_variance, noise_variance, mean_function=0.0):
        """
        @param length_scale: list/nparray containing length scales of each axis respectively
        @param signal_variance
        @param noise_variance
        @param mean_function

        Example usage: InitGP([1.5, 1.5], 1, 0.1)
        """
        self.covariance_function = SquareExponential(np.array(length_scale), signal_variance=signal_variance,
                                                     noise_variance=noise_variance)
        self.gp = GaussianProcess(self.covariance_function, mean_function=mean_function)
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
        # the empirical mean of the dataset
        # required for subtracting from measurements - gives better plotting
        self.empirical_mean = model.mean

    def InitPlanner(self, grid_domain, grid_gap, epsilon, gamma, batch_size, horizon):
        """
        Creates a planner. For now, we only allow gridded/latticed domains.

        @param grid_domain - 2-tuple of 2-tuples; Each tuple for a dimension, each containing (min, max) for that dimension
        @param grid_gap - float, resolution of our domain
        @param epsilon - maximum allowed policy loss
        @param gamma - discount factor
        @param H - int, horizon

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
        self.epsilon = epsilon
        self.gamma = gamma
        self.H = horizon
        self.batch_size = batch_size

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
        self.past_measurements = np.apply_along_axis(self.model, 1, past_locations)

    def Test(self, num_timesteps_test, method, num_samples, visualize=False, action_set=None, save_per_step=True,
             save_folder="default_results/", MCTSMaxNodes=10 ** 15):

        # history includes currrent state
        x_0 = AugmentedState(self.initial_physical_state,
                             initial_history=History(self.past_locations, self.past_measurements))
        state_history = [x_0]

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        total_reward = 0
        reward_history = []
        total_reward_history = []

        normalized_total_reward = 0
        normalized_reward_history = []
        normalized_total_reward_history = []

        base_measurement_history = []
        measurement_history = []

        total_nodes_expanded = 0
        nodes_expanded_history = []

        for time in xrange(num_timesteps_test):
            allowed_horizon = DynamicHorizon(t=time, H_max=self.H, t_max=num_timesteps_test)
            tp = TreePlan(grid_domain=self.grid_domain, grid_gap=self.grid_gap, gaussian_process=self.gp,
                          macroaction_set=action_set,
                          beta=self.beta, bad_places=self.bad_places,
                          num_samples=num_samples, batch_size=self.batch_size, horizon=allowed_horizon,
                          model=self.model)

            if method == Methods.Anytime:
                print "anytime  " + str(self.epsilon)
                bounds, x_temp_physical, nodes_expanded = tp.AnytimeAlgorithm(self.epsilon, x_0, allowed_horizon,
                                                                              max_nodes=MCTSMaxNodes)
                # TODO fix this ugly hack
                a = np.zeros(x_temp_physical.shape)
                x_temp = tp.TransitionP(x_0, a)
                x_temp.physical_state = x_temp_physical

            elif method == Methods.Exact:
                vBest, x_temp, nodes_expanded = tp.StochasticFull(x_0, allowed_horizon)

            elif method == Methods.MyopicUCB:
                vBest, x_temp, nodes_expanded = tp.StochasticFull(x_0, 1)

            elif method == Methods.MLE:
                vBest, x_temp, nodes_expanded = tp.MLE(x_0, allowed_horizon)

            elif method == Methods.qEI:
                vBest, x_temp, nodes_expanded = tp.qEI(x_0)

            else:
                raise Exception("Unknown method type")

            # x_temp is already augmented state

            # Take action a
            # x_temp = tp.TransitionP(x_0, a)
            # Draw an actual observation from the underlying environment field and add it to the our measurements

            baseline_measurements = np.asarray(
                [self.model(single_agent_state) for single_agent_state in x_temp.physical_state])

            # for simulated case, noise is incorporated into model
            """
            if self.simulate_noise_in_trials:
                noise_components = np.random.normal(0, math.sqrt(self.noise_variance), self.batch_size)
            else:
                noise_components = np.asarray([0 for i in range(self.batch_size)])
            """
            # NB shift measurements by mean
            # percieved_measurements = np.add(baseline_measurements, noise_components)
            percieved_measurements = baseline_measurements

            x_next = tp.TransitionH(x_temp, percieved_measurements)

            # Update future state
            x_0 = x_next

            reward_obtained = self.reward_function(percieved_measurements)
            normalized_reward = reward_obtained - self.batch_size * self.empirical_mean

            # Accumulated measurements
            reward_history.append(reward_obtained)
            total_reward += reward_obtained
            total_reward_history.append(total_reward)

            normalized_reward_history.append(normalized_reward)
            normalized_total_reward += normalized_reward
            normalized_total_reward_history.append(normalized_total_reward)

            measurement_history.append(percieved_measurements)
            base_measurement_history.append(baseline_measurements)
            total_nodes_expanded += nodes_expanded
            nodes_expanded_history.append(nodes_expanded)

            """
            if debug:
                print "A = ", a
                print "M = ", percieved_measurement
                print "X = "
                print "Noise = ", noise_component
                print x_0.to_str()
            """
            # Add to plot history
            state_history.append(x_0)

            if save_per_step:
                self.Visualize(state_history=state_history, display=visualize,
                               save_path=save_folder + "step" + str(time))
                # Save to file
                f = open(save_folder + "step" + str(time) + ".txt", "w")
                f.write(x_0.to_str() + "\n")
                f.write("Total accumulated reward = " + str(total_reward))
                f.close()

        # Save for the whole trial
        self.Visualize(state_history=state_history, display=visualize, save_path=save_folder + "summary")
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
        f.write(str(nodes_expanded_history) + "\n")
        f.write("Total nodes expanded = " + str(total_nodes_expanded) + "\n")

        f.write("Reward history " + str(total_reward_history) + "\n")
        f.write("Normalized Reward history " + str(normalized_total_reward_history) + "\n")
        f.close()

        """
        name_label = "test"
        result_data = []
        result_data.append([name_label, total_reward_history])
        PlotData(result_data, save_folder)
        """
        # return state_history, reward_history, nodes_expanded_history, base_measurement_history, total_reward_history
        return normalized_total_reward_history

    def Visualize(self, state_history, display=True, save_path=None):
        """ Visualize 2d environments
        """
        XGrid = np.arange(self.grid_domain[0][0], self.grid_domain[0][1] - 1e-10, self.grid_gap)
        YGrid = np.arange(self.grid_domain[1][0], self.grid_domain[1][1] - 1e-10, self.grid_gap)
        XGrid, YGrid = np.meshgrid(XGrid, YGrid)

        ground_truth = np.vectorize(lambda x, y: self.model([x, y]))
        """
        posterior_mean_before = np.vectorize(
            lambda x, y: self.gp.GPMean(locations=state_history[-2].history.locations,
                                        measurements=state_history[-2].history.measurements, current_location=[x, y]))
        posterior_mean_after = np.vectorize(
            lambda x, y: self.gp.GPMean(locations=state_history[-1].history.locations,
                                        measurements=state_history[-1].history.measurements, current_location=[x, y]))
        """
        # Plot graph of locations
        vis = Vis2d()
        vis.MapPlot(grid_extent=[self.grid_domain[0][0], self.grid_domain[0][1], self.grid_domain[1][0],
                                 self.grid_domain[1][1]],
                    ground_truth=ground_truth(XGrid, YGrid),
                    path_points=[x.physical_state for x in state_history],
                    display=display,
                    save_path=save_path)


def testWithFixedParameters(model, horizon, start_location, num_timesteps_test, method, num_samples, batch_size,
                            epsilon_=5.0,
                            save_folder=None, save_per_step=True,
                            action_set=None, MCTSMaxNodes=10 ** 15, beta=0.0):

    hyper_storer = RoadHypersStorer_Log18()

    initial_physical_state = hyper_storer.GetInitialPhysicalState(start_location)

    # print model.GenerateRoadMacroActions(initial_physical_state[-1], batch_size)

    # includes current state
    past_locations = np.array(
        [[1.0, 0.85], [1.0, 1.15], [1.15, 1.0], [0.85, 1.0], [1.0, 0.65], [1.0, 1.35], [1.35, 1.0], [0.65, 1.0],
         [1.0, 1.0]])
    past_locations = np.array(
        [[1.0, 0.5], [1.0, 1.5], [1.5, 1.0], [0.5, 1.0], [1.0, 1.0]])
    past_locations = np.array([[1.0, 1.0]])
    past_locations = np.copy(initial_physical_state)

    print "Start location " + str(past_locations) + "\n"

    # Unused
    noise_in_trials = True

    TPT = TreePlanTester(simulate_noise_in_trials=noise_in_trials, beta=beta)
    # this GP is for prediction
    TPT.InitGP(length_scale=hyper_storer.length_scale, signal_variance=hyper_storer.signal_variance,
               noise_variance=hyper_storer.noise_variance,
               mean_function=hyper_storer.mean_function)
    # adds noise to observations
    TPT.InitEnvironment(environment_noise=hyper_storer.noise_variance, model=model)
    TPT.InitPlanner(grid_domain=hyper_storer.grid_domain, grid_gap=hyper_storer.grid_gap, gamma=1, epsilon=epsilon_,
                    horizon=horizon,
                    batch_size=batch_size)
    TPT.InitTestParameters(initial_physical_state=initial_physical_state, past_locations=past_locations)

    return TPT.Test(num_timesteps_test=num_timesteps_test, visualize=False,
                    save_folder=save_folder,
                    action_set=action_set, save_per_step=save_per_step, MCTSMaxNodes=MCTSMaxNodes, method=method,
                    num_samples=num_samples)


"""
def TestRealData(locations, values, length_scale, signal_variance, noise_variance, mean_function, grid_domain,
                 start_location,
                 grid_gap=1.0, epsilon=1.0, depth=5, num_timesteps_test=20, save_folder=None, save_per_step=True,
                 MCTS=True, MCTSMaxNodes=10 ** 15, Randomized=False,
                 reward_model="Linear", sd_bonus=0.0, special=None, bad_places=[]):


    m = MapValueDict(locations, values)

    TPT = TreePlanTester(simulate_noise_in_trials=False, reward_model=reward_model, sd_bonus=sd_bonus,
                         bad_places=bad_places)
    TPT.InitGP(length_scale=length_scale, signal_variance=signal_variance, noise_variance=noise_variance,
               mean_function=mean_function)
    TPT.InitEnvironment(environment_noise=noise_variance, model=m)
    TPT.InitPlanner(grid_domain=grid_domain, grid_gap=grid_gap, gamma=1, epsilon=epsilon, H=depth)
    TPT.InitTestParameters(initial_physical_state=np.array(start_location), past_locations=np.array([start_location]))

    # For transect-type sampling
    # For normal
    return TPT.Test(num_timesteps_test=num_timesteps_test, debug=True, visualize=False, action_set=None,
                    save_folder=save_folder, save_per_step=save_per_step, MCTS=MCTS, MCTSMaxNodes=MCTSMaxNodes,
                    Randomized=Randomized, special=special)


300
"""
if __name__ == "__main__":
    # assert len(sys.argv) == 2, "Wrong number of arguments"
    print "bla"

    # Transect(seed=i)

    # print "Performing sanity checks"
    # SanityCheck()
    # print "Performing Exploratory"
    # Exploratory(1.0) # This goes to weird places
    # print "Performing Exploratory 2"
    # Exploratory(0.5)
