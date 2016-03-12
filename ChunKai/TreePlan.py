import copy
import math
import itertools

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal

from GaussianProcess import GaussianProcess
from GaussianProcess import SquareExponential
from Vis2d import Vis2d
from mutil import mutil


class TreePlan:
    """
    TODO: allow for more flexible initialization
    def __init__(self, states, actions, transition, reward, GP):
        pass
    """

    def __init__(self, grid_domain, grid_gap, gaussian_process, action_set=None, max_nodes=None, reward_type="Linear",
                 sd_bonus=0.0, bad_places=None, number_of_nodes_function=None, batch_size=1):
        """
        - Gradularity given by grid_gap
        - Squared exponential covariance function
        - Characteristic length scale the same for both directions
        """
        # size of the team
        self.batch_size = batch_size

        # Preset constants
        self.INF = 10 ** 15

        # Problem parameters
        self.grid_gap = grid_gap
        # actions available for one agent
        self.single_agent_action_set = action_set
        if action_set == None:
            self.single_agent_action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0),
                               (-grid_gap, 0))  # TODO: ensure that we can handle > 2 dimensions
        elif action_set == 'GridWithStay':
            self.single_agent_action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0), (-grid_gap, 0), (0.0, 0.0))
        self.grid_domain = grid_domain

        # an ugly way to create joint actions
        self.joint_action_set = np.asarray(list(itertools.product(self.single_agent_action_set, repeat = self.batch_size)))
        self.gp = gaussian_process

        # user defined function for number of nodes at every level
        # in the form
        # lambda t: f(t)
        # set default value
        if number_of_nodes_function is None:
            number_of_nodes_function = lambda t: 10
        self.nodes_function = number_of_nodes_function

        #TODO change into fixed reward
        if reward_type == "Linear":
            self.reward_analytical = lambda mu, sigma: mu + sd_bonus * (sigma)
            self.reward_sampled = lambda f: 0

            self.l1 = 0
            self.l2 = lambda sigma: 1
        elif reward_type == "Positive_log":
            self.reward_analytical = lambda mu, sigma: sd_bonus * (sigma)
            self.reward_sampled = lambda f: math.log(f) if f > 1 else 0.0

            self.l1 = 1
            self.l2 = lambda sigma: 0
        elif reward_type == "Step1mean":  # Step function with cutoff at 1
            self.reward_analytical = lambda mu, sigma: 1 - norm.cdf(x=1, loc=mu, scale=sigma) + sd_bonus * (sigma)
            self.reward_sampled = lambda f: 0

            self.l1 = 0
            self.l2 = lambda sigma: 1 / (math.sqrt(2 * math.pi) * sigma)
        elif reward_type == "Step15mean":  # Step function with cutoff at 1.5
            self.reward_analytical = lambda mu, sigma: 1 - norm.cdf(x=1.5, loc=mu, scale=sigma) + sd_bonus * (sigma)
            self.reward_sampled = lambda f: 0

            self.l1 = 0
            self.l2 = lambda sigma: 1 / (math.sqrt(2 * math.pi) * sigma)
        else:
            assert False, "Unknown reward type"


    def StochasticAlgorithm(self, x_0, H):
        """
         NOTE
         This implementation doesn't perform correction by introducing deterministic component
        """
        st = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H)

        x = x_0
        valid_actions = self.GetValidActionSet(x.physical_state)
        vBest = -self.INF
        aBest = valid_actions[0]

        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[a]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            q_value = self.Q_Stochastic(H, x_next, new_st) + r

            if (q_value > vBest):
                aBest = a
                vBest = q_value

        return vBest, aBest

    def V_Stochastic(self, T, x, st):

        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0

        vBest = -self.INF
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[a]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.Q_Stochastic(T, new_st) + r

            if (f > vBest):
                vBest = f

        return vBest

    def Q_Stochastic(self, T, x, new_st):
        # print "Q: p", p
        # Initialize variables
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = math.sqrt(new_st.variance)

        # the number of samples is given by user-defined function
        samples = np.random.normal(mu, sd, self.nodes_function(T))

        sample_v_values = [self.V_Stochastic(T - 1, self.TransitionH(x, sam), new_st) + self.reward_sampled(sam) for sam
                           in samples]
        avg = np.mean(sample_v_values)

        return avg

    def Preprocess(self, physical_state, locations, H):

        # physical state is a list of k points
        """
        Builds the preprocessing tree and performs the necessary precalculations
        @return root node, epsilon, lambda and number of nodes expanded required of the semi-tree built

        root node: root node of the semi tree
        epsilon: the suggested epsilon that we use so that we do not exceed the maximum number of nodes
        lambda: amount of error allowed per level
        """
        # just wrapper
        root_ss = SemiState(physical_state, locations)
        # tree
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)
        return root_node

    def BuildTree(self, node, H, isRoot=False):
        """
        Builds the preprocessing (semi) tree recursively
        """
        # history for root possibly empty
        if not isRoot: node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return

        # Add in new children for each valid action
        valid_actions = self.GetValidActionSet(node.ss.physical_state)
        for a in valid_actions:
            # Get new semi state
            cur_physical_state = node.ss.physical_state
            new_physical_state = self.PhysicalTransition(cur_physical_state, a)
            new_history_locations = np.append(node.ss.locations, cur_physical_state, 0)
            new_ss = SemiState(new_physical_state, new_history_locations)

            # Build child subtree
            new_st = SemiTree(new_ss)
            # add child to old state
            node.AddChild(a, new_st)
            # why calculate twice?
            #new_st.ComputeWeightsAndVariance(self.gp)
            self.BuildTree(new_st, H - 1)

    def GetValidActionSet(self, physical_state):
        return [a for a in self.joint_action_set if self.IsValidAction(physical_state, a)]

    def IsValidAction(self, physical_state, a):
        # TODO: ensure scalability to multiple dimensions
        # TODO: ensure epsilon comparison for floating point comparisons (currently comparing directly like a noob)
        assert physical_state.shape == a.shape
        new_state = np.add(physical_state, a)
        values = new_state.tolist()

        for i in xrange(len(values)):
            if values[i] < self.grid_domain[dim][0] or new_state[dim] >= self.grid_domain[dim][1]: return False

        return True


    # Hacks to overcome bad design
    def TransitionP(self, augmented_state, action):
        return TransitionP(augmented_state, action)

    def TransitionH(self, augmented_state, measurement):
        return TransitionH(augmented_state, measurement)

    def PhysicalTransition(self, physical_state, action):
        return PhysicalTransition(physical_state, action)



    def DeterministicML(self, x_0, H):
        """
        @param x_0 - augmented state
        @return approximately optimal value, answer, and number of node expansions
        """
        # x_0 stores a 2D np array of k points with history
        print "Preprocessing weight spaces..."
        # We take current position and past locations separately
        #st = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H)
        st = self.Preprocess(x_0.physical_state, x_0.history.locations, H)

        print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.V_ML(H, x_0, st)

        return Vapprox, Aapprox, -1

    def V_ML(self, T, x, st):
        """
        @return vBest - approximate value function computed
        @return aBest - action at the root for the policy defined by alg1
        @param st - root of the semi-tree to be used
        """

        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]

        # for every action
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            # select new state obtained by transition
            new_st = st.children[a]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.Q_ML(T, x_next, new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def Q_ML(self, T, x, new_st):
        """
        Approximates the integration step derived from alg1
        @param new_st - semi-tree at this stage
        @return - approximate value of the integral/expectation
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)
        v, _ = self.V_ML(T - 1, self.TransitionH(x, mu), new_st)
        # R_1 which is also sampled
        r_1 = self.reward_sampled(mu)

        return v + r_1
        """
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = math.sqrt(new_st.variance)

        # the number of samples is given by user-defined function
        samples = np.random.normal(mu, sd, self.nodes_function(T))

        sample_v_values = [self.V_Stochastic(T - 1, self.TransitionH(x, sam), new_st) + self.reward_sampled(sam) for sam
                           in samples]
        avg = np.mean(sample_v_values)

        return avg

### TRANSITION AND MEASUREMENTS ###

def TransitionP(augmented_state, action):
    """
    @return - copy of augmented state with physical_state updated
    """
    new_augmented_state = copy.deepcopy(augmented_state)
    new_augmented_state.physical_state = PhysicalTransition(new_augmented_state.physical_state, action)
    return new_augmented_state


def TransitionH(augmented_state, measurement):
    """
    @return - copy of augmented state with history updated
    """
    new_augmented_state = copy.deepcopy(augmented_state)
    new_augmented_state.history.append(new_augmented_state.physical_state, measurement)
    return new_augmented_state


def PhysicalTransition(physical_state, action):
    """
    @param - physical_state: numpy array with same size as action
    @return - new physical state after taking the action
    """
    # action should be joint
    new_physical_state = np.add(physical_state, action)

    return new_physical_state


# just state and history
class AugmentedState:
    def __init__(self, physical_state, initial_history):
        """
        Initialize augmented state with initial position and history
        """
        #2D array
        self.physical_state = physical_state
        #2D array
        self.history = initial_history

    def to_str(self):
        return \
            "Physical State\n" + \
            str(self.physical_state) + "\n" + \
            "Locations\n" + \
            str(self.history.locations) + "\n" + \
            "Measurements\n" + \
            str(self.history.measurements)


class SemiTree:
    def __init__(self, semi_state):
        self.ss = semi_state
        self.children = dict()
        self.weights = None  # Weight space vector
        self.variance = None  # Precomputed posterior variance

    def AddChild(self, action, semi_tree):
        self.children[action] = semi_tree

    def ComputeWeightsAndVariance(self, gp):
        self.weights, self.variance = gp.GetWeightsAndVariance(self.ss.locations, self.ss.physical_state)


class SemiState:
    """ State which only contains locations visited and its current location
    """

    def __init__(self, physical_state, locations):
        self.physical_state = physical_state
        self.locations = locations


class History:
    def __init__(self, initial_locations, initial_measurements):
        self.locations = initial_locations
        self.measurements = initial_measurements

    def append(self, new_locations, new_measurements):
        """
        new_measurements - 1D array
        new_locations - 2D array
        @modifies - self.locations, self.measurements
        """
        self.locations = np.append(self.locations, new_locations, axis=0)
        #1D array
        self.measurements = np.append(self.measurements, new_measurements)


if __name__ == "__main__":

    # Init GP: Init hyperparameters and covariance function
    length_scale = [1.5, 1.5]
    signal_variance = 1
    noise_variance = 0.1
    covariance_function = SquareExponential(np.array(length_scale), signal_variance)
    gp = GaussianProcess(covariance_function, noise_variance)

    # Init environment model
    actual_noise_variance = 0.1
    magnitude_scale = 1.0
    # model = lambda xy: magnitude_scale * multivariate_normal(mean=[0,0], cov=[[64,0],[0,64]]).pdf(xy)
    model = lambda xy: magnitude_scale * multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]).pdf(xy)

    # Planning parameters: domain and resolution
    grid_domain = ((-10, 10), (-10, 10))
    grid_gap = 0.2

    # Planning parameters:
    epsilon = 0.175  # Tolerance for policy loss
    gamma = 1.0  # Discount factor
    H = 3  # Search horizon

    # TreePlan tester
    num_timesteps_test = 20
    # Initial augmented state
    initial_physical_state = np.array([1.0, 1.0])
    initial_locations = np.array([[-1.0, -1.0], [1.0, 1.0]])
    initial_measurements = np.apply_along_axis(lambda xy: model(xy), 1, initial_locations)
    x_0 = AugmentedState(initial_physical_state,
                         initial_history=History(initial_locations, initial_measurements))

    state_history = [x_0]
    for time in xrange(num_timesteps_test):
        tp = TreePlan(grid_domain, grid_gap, gp)

        print tp.MCTSExpand(epsilon, gamma, x_0, H)

        _, a, _ = tp.DeterministicML(x_0, H)

        # Take action a
        x_temp = tp.TransitionP(x_0, a)
        # Draw an actual observation from the underlying environment field and add it to the our measurements
        measurement = model(x_temp.physical_state)
        x_next = tp.TransitionH(x_temp, measurement)

        # Update future state
        x_0 = x_next

        print "A = ", a
        print "M = ", measurement
        print "X = "
        print x_0.to_str()

        # Add to plot history
        state_history.append(x_0)

    XGrid = np.arange(grid_domain[0][0], grid_domain[0][1] + 1e-10, grid_gap)
    YGrid = np.arange(grid_domain[1][0], grid_domain[1][1] + 1e-10, grid_gap)
    XGrid, YGrid = np.meshgrid(XGrid, YGrid)
    model_grid = np.vectorize(lambda x, y: model([x, y]))
    # Plot graph of locations
    vis = Vis2d()
    vis.MapPlot(model_grid(XGrid, YGrid),  # Mesh grid
                [grid_domain[0][0], grid_domain[0][1], grid_domain[1][0], grid_domain[1][1]],
                [x.physical_state for x in state_history])
