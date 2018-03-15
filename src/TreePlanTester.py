import os

from GaussianProcess import SquareExponential, GaussianProcess
from TreePlan import *
from src.core.AugmentedState import AugmentedState
from src.core.History import History
from src.plotting.DatasetPlotGenerator import DatasetPlotGenerator
from src.enum.MethodEnum import Methods
from DynamicHorizon import DynamicHorizon


class TreePlanTester:
    def __init__(self, beta):
        self.reward_function = lambda z: sum(z)

        self.beta = beta

    # just sets the parameters
    def InitGP(self, length_scale, signal_variance, noise_variance, mean_function):

        self.covariance_function = SquareExponential(np.array(length_scale), signal_variance=signal_variance,
                                                     noise_variance=noise_variance)
        self.gp = GaussianProcess(self.covariance_function, mean_function=mean_function)
        self.noise_variance = noise_variance

    def InitEnvironment(self, environment_noise, model, hyper_storer):

        self.environment_noise = environment_noise
        self.model = model
        # the empirical mean of the dataset
        # required for subtracting from measurements - gives better plotting
        self.empirical_mean = model.mean
        self.hyper_storer = hyper_storer

    def InitPlanner(self, domain_descriptor, epsilon, gamma, batch_size, horizon):

        # self.grid_domain = grid_domain
        # self.grid_gap = grid_gap
        self.domain_descriptor = domain_descriptor
        self.epsilon = epsilon
        self.gamma = gamma
        self.H = horizon
        self.batch_size = batch_size

    def InitTestParameters(self, initial_physical_state, past_locations):
        self.initial_physical_state = initial_physical_state
        self.past_locations = past_locations

        # Compute measurements
        self.past_measurements = np.apply_along_axis(self.model, 1, past_locations)

    def Test(self, num_timesteps_test, method, num_samples, action_set=None, save_per_step=True,
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
            tp = TreePlan(domain_descriptor=self.domain_descriptor, gaussian_process=self.gp,
                          macroaction_set=action_set,
                          beta=self.beta,
                          num_samples=num_samples, batch_size=self.batch_size, horizon=allowed_horizon,
                          model=self.model)

            if method == Methods.Anytime:
                print "anytime  " + str(self.epsilon)
                bounds, x_temp_physical, nodes_expanded = tp.AnytimeAlgorithm(self.epsilon, x_0, allowed_horizon,
                                                                              max_nodes=MCTSMaxNodes)
                # TODO fix this ugly hack
                # a = np.zeros(x_temp_physical.shape)
                x_temp = TransitionP(x_0, x_temp_physical)
                # x_temp.physical_state = x_temp_physical

            elif method == Methods.Exact:
                vBest, x_temp, nodes_expanded = tp.StochasticFull(x_0, allowed_horizon)

            elif method == Methods.BucbPE:
                _, x_temp, nodes_expanded = tp.BUCB_PE(x_0, time)

            elif method == Methods.BUCB:
                _, x_temp, nodes_expanded = tp.BUCB(x_0, time)

            elif method == Methods.LP:
                _, x_temp, nodes_expanded = tp.LP(x_0, time)

            elif method == Methods.MyopicUCB:
                vBest, x_temp, nodes_expanded = tp.StochasticFull(x_0, 1)

            elif method == Methods.MLE:
                vBest, x_temp, nodes_expanded = tp.MLE(x_0, allowed_horizon)

            elif method == Methods.qEI:
                vBest, x_temp, nodes_expanded = tp.qEI(x_0)

            elif method == Methods.new_qEI:
                # R package gives error if the number of locations is less than dimension
                if time > 0:
                    vBest, x_temp, nodes_expanded = tp.new_qEI(x_0)
                else:
                    vBest, x_temp, nodes_expanded = tp.qEI(x_0)

            elif method == Methods.EI:
                vBest, x_temp, nodes_expanded = tp.EI(x_0)

            elif method == Methods.PI:
                vBest, x_temp, nodes_expanded = tp.PI(x_0)

            else:
                raise Exception("Unknown method type")

            # x_temp is already augmented state

            # Take action a
            # x_temp = tp.TransitionP(x_0, a)
            # Draw an actual observation from the underlying environment field and add it to the our measurements

            baseline_measurements = np.asarray(
                [self.model(single_agent_state) for single_agent_state in x_temp.physical_state])

            # NB shift measurements by mean
            # percieved_measurements = np.add(baseline_measurements, noise_components)
            percieved_measurements = baseline_measurements

            x_next = TransitionH(x_temp, percieved_measurements)

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

            # Add to plot history
            state_history.append(x_0)

            if save_per_step:
                self.Visualize(state_history=state_history,
                               save_path=save_folder + "step" + str(time))
                # Save to file
                f = open(save_folder + "step" + str(time) + ".txt", "w")
                f.write(x_0.to_str() + "\n")
                f.write("Total accumulated reward = " + str(total_reward) + '\n')
                f.write("Nodes expanded = " + str(nodes_expanded))
                f.close()

        # Save for the whole trial
        self.Visualize(state_history=state_history, save_path=save_folder + "summary")
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

        self.hyper_storer.PrintParamsToFile(save_folder + "hypers_used.txt")

        # return state_history, reward_history, nodes_expanded_history, base_measurement_history, total_reward_history
        return normalized_total_reward_history

    def Visualize(self, state_history, save_path):

        plot_generator = DatasetPlotGenerator(self.model.dataset_type)

        plot_generator.GeneratePlot(model=self.model,
                                    path_points=[x.physical_state for x in state_history],
                                    save_path=save_path)


def testWithFixedParameters(model, horizon, num_timesteps_test, method, num_samples,
                            epsilon_=5.0,
                            save_folder=None, save_per_step=True,
                            action_set=None, MCTSMaxNodes=10 ** 15, beta=0.0):
    """
    if time_slot == 44:
        hyper_storer = RoadHypersStorer_Log44()
    elif time_slot == 18:
        hyper_storer = RoadHypersStorer_Log18()
    else:
        raise Exception("wrong tzxi time slot")
    """

    # hyper_storer = RoadHypersStorer_18()
    hyper_storer = model.hyper_storer
    initial_physical_state = model.start_location


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

    TPT = TreePlanTester(beta=beta)
    # this GP is for prediction
    TPT.InitGP(length_scale=hyper_storer.length_scale, signal_variance=hyper_storer.signal_variance,
               noise_variance=hyper_storer.noise_variance,
               mean_function=hyper_storer.mean_function)
    # adds noise to observations
    TPT.InitEnvironment(environment_noise=hyper_storer.noise_variance, model=model, hyper_storer=hyper_storer)
    TPT.InitPlanner(domain_descriptor=model.domain_descriptor, gamma=1, epsilon=epsilon_,
                    horizon=horizon,
                    batch_size=model.batch_size)
    TPT.InitTestParameters(initial_physical_state=initial_physical_state, past_locations=past_locations)

    return TPT.Test(num_timesteps_test=num_timesteps_test,
                    save_folder=save_folder,
                    action_set=action_set, save_per_step=save_per_step, MCTSMaxNodes=MCTSMaxNodes, method=method,
                    num_samples=num_samples)
