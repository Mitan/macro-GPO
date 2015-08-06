"""

Implements Algorithm I in my CA report

- iterative Deepening
- epsilon optimal

"""

import numpy as np
from scipy import linalg
from scipy.stats import norm
from matplotlib import pyplot as pl
from GaussianProcess import GaussianProcess
from GaussianProcess import SquareExponential
from Vis2d import Vis2d
from scipy.stats import multivariate_normal
from mutil import mutil

import copy
import sys
import math


class TreePlan:
    """
	TODO: allow for more flexible initialization
	def __init__(self, states, actions, transition, reward, GP):
		pass
	"""

    static_mathutil = mutil()
    static_mathutil.Init(200)
    # static_mathutil.Init(25000)

    def __init__(self, grid_domain, grid_gap, gaussian_process, action_set=None, max_nodes=None, reward_type="Linear",
                 sd_bonus=0.0, bad_places=None):
        """
		- Gradularity given by grid_gap
		- Squared exponential covariance function
		- Characteristic length scale the same for both directions
		"""

        # Preset constants
        self.INF = 10 ** 15
        self.PEWPEW = 0.5  # 1.0-1.0/10000.0

        # Problem parameters
        self.grid_gap = grid_gap
        self.action_set = action_set
        if action_set == None:
            self.action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0),
                               (-grid_gap, 0))  # TODO: ensure that we can handle > 2 dimensions
        elif action_set == 'GridWithStay':
            self.action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0), (-grid_gap, 0), (0.0, 0.0))
        self.grid_domain = grid_domain
        self.gp = gaussian_process
        self.max_nodes = self.INF if max_nodes == None else max_nodes
        self.sd_bonus = sd_bonus

        # Obstacles
        self.bad_places = bad_places

        # Precomputed algo stuff
        # TODO: factor this outside, or pass as a parameter to TreePlan. We don't need to keep recomputing!
        self.mathutil = TreePlan.static_mathutil

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

    def EI(self, x_0):

        """
		*Myopic* implementation of EI (Expected improvement)
		"""

        best_observation = max(x_0.history.measurements)
        best_action = None
        best_expected_improv = 0.0

        valid_actions = self.GetValidActionSet(x_0.physical_state)

        for a in valid_actions:
            x_next = self.TransitionP(x_0, a)

            chol = self.gp.GPCholTraining(x_0.history.locations)
            cov_query = self.gp.GPCovQuery(x_0.history.locations, x_next.physical_state)
            weights = self.gp.GPWeights(x_0.history.locations, x_next.physical_state, chol, cov_query)
            var = self.gp.GPVariance2(x_0.history.locations, x_next.physical_state, chol, cov_query)

            sigma = math.sqrt(var)
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=weights)

            Z = (mean - best_observation) / sigma
            expectedImprov = (mean - best_observation) * norm.cdf(x=Z, loc=0, scale=1.0) + sigma * norm.pdf(x=Z, loc=0,
                                                                                                            scale=1.0)

            if expectedImprov >= best_expected_improv:
                best_expected_improv = expectedImprov
                best_action = a

        return best_expected_improv, best_action, len(valid_actions)

    def PI(self, x_0):

        """
		*Myopic implementation of PI (probability of improvement)*
		"""

        best_observation = max(x_0.history.measurements)
        best_action = None
        best_improv_prob = 0.0

        valid_actions = self.GetValidActionSet(x_0.physical_state)

        for a in valid_actions:
            x_next = self.TransitionP(x_0, a)

            chol = self.gp.GPCholTraining(x_0.history.locations)
            cov_query = self.gp.GPCovQuery(x_0.history.locations, x_next.physical_state)
            weights = self.gp.GPWeights(x_0.history.locations, x_next.physical_state, chol, cov_query)
            var = self.gp.GPVariance2(x_0.history.locations, x_next.physical_state, chol, cov_query)

            sigma = math.sqrt(var)
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=weights)

            improv_prob = 1.0 - norm.cdf(x=best_observation, loc=mean, scale=sigma)

            if improv_prob >= best_improv_prob:
                best_improv_prob = improv_prob
                best_action = a

        return best_improv_prob, best_action, len(valid_actions)

    def Algorithm1(self, epsilon, x_0, H):
        """
		@param x_0 - augmented state
		@return approximately optimal value, answer, and number of node expansions
		"""

        # Obtain lambda
        # l = epsilon / (gamma * H * (H+1))
        # l = epsilon / sum([gamma ** i for i in xrange(1, H+1)])

        print "Preprocessing weight spaces..."
        # Obtain Lipchitz lookup tree
        st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,
                                                             epsilon)
        print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.EstimateV(H, l, x_0, st)

        return Vapprox, Aapprox, nodes_expanded

    def RandomSampling(self, epsilon, x_0, H):
        #print epsilon
        st, _, __, ___ = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H, epsilon)
        beta = epsilon / H
        e_s = beta / 4

        for T in reversed(range(H)):
            max_err = self.FindMLEError(st)
            delta = min(beta / 8 / max_err, 1)
            lamb = e_s / H
            # print "lambda is " + str( lamb)
            x = x_0

            # set of valid actions
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
                future_reward = self.EstimateQ(T, lamb, x_next, new_st, True) + r  # using MLE
                future_reward_random = self.ComputeQRandom(T, lamb, x_next, 1.0 - delta, new_st) + r

                # Correction step
                qmod = future_reward_random
                if abs(future_reward_random - future_reward) > e_s + max_err:
                    qmod = future_reward

                if (qmod > vBest):
                    aBest = a
                    vBest = qmod
        return vBest, aBest, 0

    def ComputeVRandom(self, T, l, x, p, st):

        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0, None

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
            f = self.ComputeQRandom(T, l, x_next, p ** (1.0 / len(valid_actions)), new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def ComputeQRandom(self, T, l, x, p, new_st):

        # patological case. perhaps should work like this
        if T == 0:
            return 0
        #print x.physical_state, T
        # print "Q: p", p
        # Initialize variables
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)
        sd = math.sqrt(new_st.variance)

        # idk wtf is pewpew but let it just stay
        a = math.log(0.5 - 0.5 * (p ** self.PEWPEW))
        b = (self.l1 + new_st.lipchitz) ** 2
        c = (l ** 2)
        n = math.ceil(
            2 * a * b * new_st.variance / (
            - c))
        n = max(n, 1)
        #print new_st.variance, c, n
        samples = np.random.normal(mu, sd, n)
        #print len(samples)

        values_list = [self.ComputeVRandom(T - 1, l, self.TransitionH(x, sam), p ** ((1 - self.PEWPEW) / n),
                                   new_st)[0] + self.reward_sampled(sam) for sam in samples]
        avg = np.mean(values_list)

        return avg

    def FindMLEError(self, st):

        if not st.children: return 0;

        max_err = 0
        for a, semi_child in st.children.iteritems():
            new_err = math.sqrt(2 / math.pi) * (semi_child.lipchitz + self.l1) * math.sqrt(semi_child.variance)
            rec_err = self.FindMLEError(semi_child)
            max_err = max(max_err, new_err + rec_err)

        return max_err

    def Preprocess(self, physical_state, locations, H, suggested_epsilon):
        """
		Builds the preprocessing tree and performs the necessary precalculations
		@return root node, epsilon, lambda and number of nodes expanded required of the semi-tree built

		root node: root node of the semi tree
		epsilon: the suggested epsilon that we use so that we do not exceed the maximum number of nodes
		lambda: amount of error allowed per level
		"""

        root_ss = SemiState(physical_state, locations)
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)
        self.PreprocessLipchitz(root_node, isRoot=True)

        # Search for appropriate epsilon and adjust partitions accordingly
        ep = suggested_epsilon
        while True:
            l = ep / H  #
            num_nodes = self.PreprocessPartitions(root_node, l, True)
            if num_nodes > self.max_nodes:
                ep = ep * 1.1  # TODO: more accurately choose this (non-exponential growth)
                print "Trying ep: %f" % ep
            else:
                break

        print "Suggested epsilon=%f, Using epsilon=%f, num_nodes=%f" % (suggested_epsilon, ep, num_nodes)
        return root_node, ep, l, num_nodes

    def BuildTree(self, node, H, isRoot=False):
        """
		Builds the preprocessing (semi) tree recursively
		"""

        if not isRoot: node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return
        cur_physical_state = node.ss.physical_state
        # Add in new children for each valid action
        valid_actions = self.GetValidActionSet(node.ss.physical_state)
        for a in valid_actions:
            # Get new semi state
            new_physical_state = self.PhysicalTransition(cur_physical_state, a)
            new_locations = np.append(node.ss.locations, np.atleast_2d(cur_physical_state), 0)
            new_ss = SemiState(new_physical_state, new_locations)

            # Build child subtree
            new_st = SemiTree(new_ss)
            node.AddChild(a, new_st)
            new_st.ComputeWeightsAndVariance(self.gp)
            self.BuildTree(new_st, H - 1)

    def GetValidActionSet(self, physical_state):
        return [a for a in self.action_set if self.IsValidAction(physical_state, a)]

    def IsValidAction(self, physical_state, a):
        # TODO: ensure scalability to multiple dimensions
        # TODO: ensure epsilon comparison for floating point comparisons (currently comparing directly like a noob)

        new_state = physical_state + a
        ndims = 2
        for dim in xrange(ndims):
            if new_state[dim] < self.grid_domain[dim][0] or new_state[dim] >= self.grid_domain[dim][1]: return False

        # Check for obstacles
        if self.bad_places:
            for i in xrange(len(self.bad_places)):
                if abs(new_state[0] - self.bad_places[i][0]) < 0.001 and abs(
                                new_state[1] - self.bad_places[i][1]) < 0.001:
                    return False

        return True

    def PreprocessLipchitz(self, node, isRoot=False):
        """
		Obtain Lipchitz vector and Lipchitz constant for each node.
		@param node - root node of the semi-tree (assumed to be already constructed)
		"""

        nl = node.ss.locations.shape[
            0]  # number of elements PRIOR to adding current location. Ie. the size of the weight space

        # Base case
        if len(node.children) == 0:
            node.L_upper = np.zeros((nl + 1, 1))

        else:
            # Recursive case
            vmax = np.zeros((nl + 1, 1))
            for a, c in node.children.iteritems():
                self.PreprocessLipchitz(c)
                av = (
                c.L_upper[0:nl + 1] + ((c.L_upper[-1] + self.l1 + self.l2(math.sqrt(c.variance))) * (c.weights.T)))
                vmax = np.maximum(vmax, av)

            node.L_upper = vmax

        node.lipchitz = node.L_upper[-1]

    def PreprocessPartitions(self, node, err_allowed, isRoot=False):
        """
		Computes number of partitions and k for stated node and all its descendents.
		@param node - root node of the semi-tree. Assumed to have obtained variance and kxi. 
		@return number of node expansions required for this node and all other 
		"""

        if not isRoot:
            if node.lipchitz == 0 and self.l1 == 0:
                normalized_error = float('inf')
            else:
                normalized_error = err_allowed / (node.lipchitz + self.l1) / math.sqrt(node.variance)
            num_partitions, k, true_error_bound = self.mathutil.FindSmallestPartition(normalized_error)

            # Save to persistent
            node.n = num_partitions
            node.k = k
            node.true_error = (node.lipchitz + self.l1) * math.sqrt(node.variance) * true_error_bound
            assert node.true_error <= err_allowed
        else:
            # Dummy numbers that don't matter...
            node.n = 0
            node.k = 1

        # print node.n
        # if not node.n==0: print node.n+2, num_partitions

        if node.n == 0:
            translated = 1  # special case where we may double count both tail evaluations
        else:
            translated = node.n + 2  # taking into account both tail evaluations (2 of them)
        if len(node.children) == 0: return translated

        t = 0  # number of nodes that all descendents will have to expand (all the way to leaves)
        for a, c in node.children.iteritems():
            t += self.PreprocessPartitions(c, err_allowed)

        else:
            return t * translated + translated

    def EstimateV(self, T, l, x, st):
        """
		@return vBest - approximate value function computed
		@return aBest - action at the root for the policy defined by alg1
		@param st - root of the semi-tree to be used
		"""

        #print "from V " + str(T)
        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = None
            try:
                new_st = st.children[a]
            except:

                print a, st.ss.physical_state, st.children, T
            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.EstimateQ(T, l, x_next, new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def EstimateQ(self, T, l, x, new_st, isMLE = False):

        """
		Approximates the integration step derived from alg1
		@param new_st - semi-tree at this stage
		@return - approximate value of the integral/expectation
		"""

        # patological case. perhaps should work like this
        if T == 0:
            return 0

        #print "from Q " + str(T)
        # if T > 3: print T
        # Initialize variables

        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        #todo check
        if isMLE:
            return self.EstimateV(T - 1, l, self.TransitionH(x, mu), new_st)[0] + self.reward_sampled(mu)

        sd = math.sqrt(new_st.variance)
        k = new_st.k

        #  hack
        num_partitions = new_st.n + 2  # number of partitions INCLUDING the tail ends

        if new_st.n > 0: width = 2.0 * k * sd / new_st.n
        vAccum = 0

        testsum = 0
        for i in xrange(2, num_partitions):
            # Compute boundary points
            zLower = mu - sd * k + (i - 2) * width
            zUpper = mu - sd * k + (i - 1) * width

            # Compute evaluation points
            zPoints = 0.5 * (zLower + zUpper)
            v, _ = self.EstimateV(T - 1, l, self.TransitionH(x, zPoints), new_st)
            v += self.reward_sampled(zPoints)

            # Recursively compute values
            vAccum += v * (norm.cdf(x=zUpper, loc=mu, scale=sd) - norm.cdf(x=zLower, loc=mu, scale=sd))
            testsum += (norm.cdf(x=zUpper, loc=mu, scale=sd) - norm.cdf(x=zLower, loc=mu, scale=sd))

        # Weight values
        rightLimit = mu + k * sd
        leftLimit = mu - k * sd
        vRightTailVal, _ = self.EstimateV(T - 1, l, self.TransitionH(x, rightLimit), new_st)
        vRightTailVal += self.reward_sampled(rightLimit)
        if num_partitions == 2:
            vLeftTailVal = vRightTailVal  # Quick hack for cases where the algo only requires us to us MLE (don't need to repeat measurement at mean of gaussian pdf)
        else:
            vLeftTailVal, _ = self.EstimateV(T - 1, l, self.TransitionH(x, leftLimit), new_st)  # usual case
            vLeftTailVal += self.reward_sampled(leftLimit)
        vAccum += vRightTailVal * (1 - norm.cdf(x=rightLimit, loc=mu, scale=sd)) + \
                  vLeftTailVal * norm.cdf(x=leftLimit, loc=mu, scale=sd)
        # test that sum of weight is 1
        testsum += (1 - norm.cdf(x=rightLimit, loc=mu, scale=sd)) + norm.cdf(x=leftLimit, loc=mu, scale=sd)
        assert abs(testsum - 1.0) < 0.0001, "Area != 1, %f instead with %f partitions" % (testsum, num_partitions)

        return vAccum


    # Hacks to overcome bad design
    def TransitionP(self, augmented_state, action):
        return TransitionP(augmented_state, action)

    def TransitionH(self, augmented_state, measurement):
        return TransitionH(augmented_state, measurement)

    def PhysicalTransition(self, physical_state, action):
        return PhysicalTransition(physical_state, action)


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

    new_physical_state = physical_state + action

    return new_physical_state


class AugmentedState:
    def __init__(self, physical_state, initial_history):
        """
		Initialize augmented state with initial position and history
		"""
        self.physical_state = physical_state
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
        self.L_upper = None  # Upper bound for the lipchitz VECTOR L
        self.lipchitz = None  # Lipchitz constant for the next observation

    def AddChild(self, action, semi_tree):
        self.children[action] = semi_tree

    def ComputeWeightsAndVariance(self, gp):
        """ Compute the weights for this semi_state ONLY"""
        chol = gp.GPCholTraining(self.ss.locations)
        cov_query = gp.GPCovQuery(self.ss.locations, self.ss.physical_state)
        self.weights = gp.GPWeights(self.ss.locations, self.ss.physical_state, chol, cov_query)
        self.variance = gp.GPVariance2(self.ss.locations, self.ss.physical_state, chol, cov_query)


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

    def append(self, new_location, new_measurement):
        """
		@modifies - self.locations, self.measurements
		"""
        self.locations = np.append(self.locations, np.atleast_2d(new_location), axis=0)
        self.measurements = np.append(self.measurements, new_measurement)


if __name__ == "__main__":
    print list(reversed(range(3)))
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
    epsilon = 10.0  # Tolerance for policy loss
    #gamma = 1.0  # Discount factor
    H = 3  # Search horizon

    # TreePlan tester
    num_timesteps_test = 10
    # Initial augmented state
    initial_physical_state = np.array([1.0, 1.0])
    initial_locations = np.array([[-1.0, -1.0], [1.0, 1.0]])
    initial_measurements = np.apply_along_axis(lambda xy: model(xy), 1, initial_locations)
    x_0 = AugmentedState(initial_physical_state,
                         initial_history=History(initial_locations, initial_measurements))

    state_history = [x_0]
    for time in xrange(num_timesteps_test):
        tp = TreePlan(grid_domain, grid_gap, gp)

        # print tp.MCTSExpand(epsilon, gamma, x_0, H)

        _, a, _ = tp.RandomSampling(epsilon, x_0, H)

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
