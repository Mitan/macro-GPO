import os

from gp.GaussianProcess import GaussianProcess
from TreePlan import *
from src.core.AugmentedState import AugmentedState
from src.core.History import History
from src.gp.covariance.CovarianceGenerator import CovarianceGenerator
from src.plotting.DatasetPlotGenerator import DatasetPlotGenerator
from src.enum.MethodEnum import Methods
from Utils import DynamicHorizon


class TreePlanTester:
    def __init__(self, beta):
        self.reward_function = lambda z: sum(z)

        self.beta = beta

    # just sets the parameters
    def InitGP(self, hyper_storer):

        covariance_generator = CovarianceGenerator(hyper_storer)
        self.covariance_function = covariance_generator.get_covariance()
        # self.covariance_function = SquareExponential(np.array(length_scale), signal_variance=signal_variance)

        self.gp = GaussianProcess(covariance_function=self.covariance_function,
                                  mean_function=hyper_storer.mean_function,
                                  noise_variance=hyper_storer.noise_variance)
        self.noise_variance = hyper_storer.noise_variance

    def InitEnvironment(self, model, hyper_storer):

        self.model = model
        # the empirical mean of the dataset
        # required for subtracting from measurements - gives better plotting
        self.empirical_mean = model.empirical_mean
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

    def Test(self, total_budget, method, num_samples, anytime_num_iterations, action_set=None, save_per_step=True,
             save_folder="default_results/", MCTSMaxNodes=10 ** 15):

        num_time_steps = total_budget / self.batch_size
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

        for time in xrange(num_time_steps):
            allowed_horizon = DynamicHorizon(t=time, H_max=self.H, t_max=num_time_steps)
            tp = TreePlan(domain_descriptor=self.domain_descriptor, gaussian_process=self.gp,
                          macroaction_set=action_set,
                          beta=self.beta,
                          num_samples=num_samples, batch_size=self.batch_size, horizon=allowed_horizon,
                          model=self.model)

            if method == Methods.Anytime:
                print "anytime  " + str(self.epsilon)
                bounds, x_temp_physical, nodes_expanded = tp.AnytimeAlgorithm(epsilon=self.epsilon,
                                                                              x_0=x_0,
                                                                              anytime_num_iterations=anytime_num_iterations,
                                                                              H=allowed_horizon,
                                                                              max_nodes=MCTSMaxNodes)
                x_temp = TransitionP(x_0, x_temp_physical)

            elif method == Methods.Exact:
                vBest, x_temp, nodes_expanded = tp.StochasticFull(x_0, allowed_horizon)

            else:
                raise Exception("Unknown method type")

            # x_temp is already augmented state

            baseline_measurements = np.asarray(
                [self.model(single_agent_state) for single_agent_state in x_temp.physical_state])

            # NB shift measurements by mean
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

        return normalized_total_reward_history

    def Visualize(self, state_history, save_path):

        plot_generator = DatasetPlotGenerator(self.model.dataset_type)

        plot_generator.GeneratePlot(model=self.model,
                                    path_points=[x.physical_state for x in state_history],
                                    save_path=save_path)


def testWithFixedParameters(model, horizon, total_budget, method, num_samples,
                            epsilon_=5.0,
                            save_folder=None, save_per_step=True,
                            action_set=None, MCTSMaxNodes=10 ** 15, beta=0.0, anytime_num_iterations=None):

    hyper_storer = model.hyper_storer
    initial_physical_state = model.start_location

    past_locations = np.copy(initial_physical_state)

    print "Start location " + str(past_locations) + "\n"

    TPT = TreePlanTester(beta=beta)
    # this GP is for prediction
    TPT.InitGP(hyper_storer=hyper_storer)

    TPT.InitEnvironment(model=model, hyper_storer=hyper_storer)
    TPT.InitPlanner(domain_descriptor=model.domain_descriptor, gamma=1, epsilon=epsilon_,
                    horizon=horizon,
                    batch_size=model.batch_size)
    TPT.InitTestParameters(initial_physical_state=initial_physical_state, past_locations=past_locations)

    return TPT.Test(total_budget=total_budget,
                    save_folder=save_folder,
                    action_set=action_set, save_per_step=save_per_step, MCTSMaxNodes=MCTSMaxNodes, method=method,
                    num_samples=num_samples,
                    anytime_num_iterations=anytime_num_iterations)
