import copy
import math

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

from GaussianProcess import GaussianProcess
from GaussianProcess import SquareExponential
from Vis2d import Vis2d
# from mutil import mutil
from MacroActionGenerator import GenerateSimpleMacroactions
from qEI import qEI
from SampleFunctionBuilder import GetNumberOfSamples


class TreePlan:
    """
    TODO: allow for more flexible initialization
    def __init__(self, states, actions, transition, reward, GP):
        pass
    """

    def __init__(self, batch_size, grid_domain, horizon, grid_gap, num_samples, gaussian_process, model,
                 macroaction_set=None,
                 max_nodes=None,
                 beta=0.0, bad_places=None):
        """
        - Gradularity given by grid_gap
        - Squared exponential covariance function
        - Characteristic length scale the same for both directions
        """
        self.model = model

        self.batch_size = batch_size
        # Preset constants
        # self.INF = 10 ** 15

        self.H = horizon
        # Number of observations/samples generated for every node
        self.samples_per_stage = num_samples

        # Problem parameters
        self.grid_gap = grid_gap
        self.macroaction_set = macroaction_set
        if macroaction_set is None:
            self.macroaction_set = GenerateSimpleMacroactions(self.batch_size, self.grid_gap)

        self.grid_domain = grid_domain
        self.gp = gaussian_process
        self.max_nodes = float("inf") if max_nodes is None else max_nodes
        self.beta = beta

        # Obstacles
        self.bad_places = bad_places

        # Precomputed algo stuff
        # unused
        # self.mathutil = TreePlan.static_mathutil
        # unused
        self.l1 = 0
        # unused
        self.l2 = lambda sigma: 1

        self.reward_analytical = lambda mu, sigma: self.AcquizitionFunction(mu, sigma)
        # unused
        self.reward_sampled = lambda f: 0

    # heuristic
    # we use batch UCB version from Erik
    # todo check that we do not add noise twice
    def AcquizitionFunction(self, mu, sigma):
        if self.beta == 0:
            return np.sum(mu)
        exploration_matrix = np.identity(sigma.shape[0]) * (self.gp.covariance_function.noise_variance) + sigma
        return np.sum(mu) + self.beta * math.log(np.linalg.det(exploration_matrix))

    """
    def Algorithm1(self, epsilon, gamma, x_0, H):

                @param x_0 - augmented state
                @return approximately optimal value, answer, and number of node expansions


        # Obtain lambda
        # l = epsilon / (gamma * H * (H+1))
        # l = epsilon / sum([gamma ** i for i in xrange(1, H+1)])

        print "Preprocessing weight spaces..."
        # Obtain Lipchitz lookup tree
        st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state,
                                                             x_0.history.locations[: -self.batch_size, :], H,
                                                             epsilon)

        print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.EstimateV(H, l, gamma, x_0, st)

        return Vapprox, Aapprox, nodes_expanded

        return Vapprox, Aapprox, nodes_expanded
        """

    def MLE(self, x_0, H):
        """
                @param x_0 - augmented state
                @return approximately optimal value, answer, and number of node expansions
        """

        # by default physical state length is self.batch_size
        # but for the first step it is equal to 1, since it is just agent's position

        # x_0.history.locations includes current location

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        # root_ss = SemiState(x_0.physical_state, x_0.history.locations[: -self.batch_size, :])
        root_ss = SemiState(physical_state, past_locations)

        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        # st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,epsilon)

        # print "Performing search..."
        # Get answer
        Vapprox, Xapprox = self.ComputeVMLE(H, x_0, root_node)
        if math.isinf(Vapprox):
            raise Exception("MLE could not move from  location " + str(x_0.physical_state))

        return Vapprox, Xapprox, -1

    def GetNextAugmentedStates(self, current_augmented_state):
        #TODO for road create a fake action a = [0. 0. 0.] and then manually
        #TODO fix bad design
        # set physical state for every new augmented state
        # or create a new function TransitionP for setting augmented state

        # this is for real-world
        current_location = current_augmented_state.physical_state[-1, :]
        # print "H = " + str(T)
        # print current_location

        new_physical_states = self.model.GenerateRoadMacroActions(current_location, self.batch_size)

        # if we can't move further in this direction, then stop and return
        """
        if not new_physical_states:
            return -float("inf"), np.zeros((self.batch_size, 2))
        """
        if not new_physical_states:
            return []

        # print new_physical_states
        fake_action = np.zeros(new_physical_states[0].shape)
        next_states = []

        for next_p_state in new_physical_states:
            next_st = self.TransitionP(current_augmented_state, fake_action)
            next_st.physical_state = next_p_state
            next_states.append(next_st)

        return next_states

    def ComputeVMLE(self, T, x, st):

        """
                @return vBest - approximate value function computed
                @return aBest - action at the root for the policy defined by alg1
                @param st - root of the semi-tree to be used
                """
        # not needed
        if T == 0: return 0, np.zeros((self.batch_size, 2))

        # for simulated
        # valid_actions = self.GetValidActionSet(x.physical_state)
        # these are augmented states
        # next_states = [self.TransitionP(x, a) for a in valid_actions]

        next_states = self.GetNextAugmentedStates(x)
        if not next_states:
            return -float("inf"), np.zeros((self.batch_size, 2))

        vBest = - float("inf")
        xBest = next_states[0]
        for x_next  in next_states:

            # x_next = self.TransitionP(x, a)
            next_physical_state = x_next.physical_state

            # cannot tuple augmented state, need to use physical state here
            new_st = st.children[ToTuple(next_physical_state)]

            mean = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)

            var = new_st.variance
            r = self.reward_analytical(mean, var)

            # Future reward
            f = self.ComputeQMLE(T, x_next, new_st) + r

            if f > vBest:
                xBest = x_next
                vBest = f
        # xBest is augmented state
        return vBest, xBest

    def ComputeQMLE(self, T, x, new_st):
        # no need to average over zeroes
        if T == 1:
            return 0

        mu = self.gp.GPMean(measurements=x.history.measurements, weights=new_st.weights)
        # mu = self.gp.GPMean(locations=x.history.locations, measurements=x.history.measurements, current_location=x.physical_state, cholesky=new_st.cholesky)
        return self.ComputeVMLE(T - 1, self.TransitionH(x, mu), new_st)[0]

    def qEI(self, x_0, eps=10 ** (-5)):
        # x_0 stores a 2D np array of k points with history
        max_measurement = max(x_0.history.measurements)

        best_action = None
        best_expected_improv = -1.0

        # valid_actions = self.GetValidActionSet(x_0.physical_state)
        # next_states = [self.TransitionP(x_0, a) for a in valid_actions]

        next_states = self.GetNextAugmentedStates(x_0)
        if not next_states:
            raise Exception("qEI could not move from   location " + str(x_0.physical_state))

        chol = self.gp.Cholesky(x_0.history.locations)

        for x_next in next_states:
            # x_next = self.TransitionP(x_0, a)

            Sigma = self.gp.GPVariance(locations=x_0.history.locations, current_location=x_next.physical_state,
                                       cholesky=chol)
            weights = self.gp.GPWeights(locations=x_0.history.locations, current_location=x_next.physical_state,
                                        cholesky=chol)
            mu = self.gp.GPMean(measurements=x_next.history.measurements, weights=weights)

            expectedImprov = qEI(Sigma, eps, mu, max_measurement, self.batch_size)

            # comparison
            if expectedImprov >= best_expected_improv:
                best_expected_improv = expectedImprov
                best_action = x_next

        return best_expected_improv, best_action, len(next_states)

    def StochasticFull(self, x_0, H):
        # by default physical state length is self.batch_size
        # but for the first step it is equal to 1, since it is just agent's position

        # x_0.history.locations includes current location

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        root_ss = SemiState(physical_state, past_locations)
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        # st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,epsilon)

        # print "Performing search..."
        # Get answer
        Vapprox, Xapprox = self.ComputeVRandom(H, x_0, root_node)
        if math.isinf(Vapprox):
            raise Exception("Exact full H = " + str(H) + " could not move from  location " + str(x_0.physical_state))

        return Vapprox, Xapprox, -1

    def ComputeVRandom(self, T, x, st):

        # simulated
        # valid_actions = self.GetValidActionSet(x.physical_state)
        # next_states = [self.TransitionP(x, a) for a in valid_actions]

        # not needed
        if T == 0: return 0, np.zeros((self.batch_size, 2))

        next_states = self.GetNextAugmentedStates(x)
        if not next_states:
            return -float("inf"), np.zeros((self.batch_size, 2))

        vBest = -float("inf")
        xBest = next_states[0]
        for x_next in next_states:

            # x_next = self.TransitionP(x, a)
            # go down the semitree node
            next_physical_state = x_next.physical_state
            new_st = st.children[ToTuple(next_physical_state)]

            # Reward is just the mean added to a multiple of the variance at that point

            mean = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)

            # mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state, weights=new_st.weights)
            var = new_st.variance
            # print np.linalg.det(var)
            r = self.reward_analytical(mean, var)

            # Future reward
            f = self.ComputeQRandom(T, x_next, new_st) + r

            if f > vBest:
                xBest = x_next
                vBest = f

        return vBest, xBest

    def ComputeQRandom(self, T, x, new_st):

        # sams = np.random.normal(mu, sd, self.samples_per_stage)

        # no need to average over zeroes
        if T == 1:
            return 0

        mu = self.gp.GPMean(measurements=x.history.measurements, weights=new_st.weights)

        # mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = new_st.variance

        number_of_samples = GetNumberOfSamples(self.H, T)
        # sams = np.random.multivariate_normal(mu, sd, self.samples_per_stage)
        sams = np.random.multivariate_normal(mu, sd, number_of_samples)

        rrr = [self.ComputeVRandom(T - 1, self.TransitionH(x, sam),
                                   new_st)[0] for sam in sams]
        avg = np.mean(rrr)

        return avg

    """
    def NewStochasticFull(self, x_0, H):
        Vapprox, Aapprox = self.ComputeNewVRandom(H, x_0)

        return Vapprox, Aapprox, -1

    def ComputeNewVRandom(self, T, x):




        valid_actions = self.GetValidActionSet(x.physical_state)
        # not needed
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]

        locations = x.history.locations
        measurements = x.history.measurements

        lengthscale = (0.1, 0.1)
        signalvariance = 1.0
        noisevariance = 0.01

        kernel = GPy.kern.RBF(input_dim=2, variance=signalvariance, lengthscale=lengthscale,
                              ARD=True)

        measurements_2d = np.atleast_2d(measurements)
        if measurements_2d.shape[0] == 1:
            measurements_2d = measurements_2d.T

        m = GPy.models.GPRegression(X=locations, Y=measurements_2d, kernel=kernel, noise_var=noisevariance,
                                    normalizer=False)

        for a in valid_actions:

            # just physical state updated
            x_next = self.TransitionP(x, a)
            new_state = x_next.physical_state

            # go down the semitree node
            # new_st = st.children[ToTuple(a)]
            assert np.array_equal(x_next.history.locations, locations)
            assert np.array_equal(x_next.history.measurements, measurements)

            mean, var = m.predict(new_state, full_cov=True)
            # Reward is just the mean added to a multiple of the variance at that point

            # mean = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)

            # mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state, weights=new_st.weights)
            # var = new_st.variance
            # print np.linalg.det(var)
            r = self.reward_analytical(mean, var)

            # Future reward
            if T == 1:
                f = r
            else:
                f = self.ComputeNewQRandom(T, x_next, mean.flatten(), var) + r

            if f > vBest:
                aBest = a
                vBest = f

        return vBest, aBest

    def ComputeNewQRandom(self, T, x, mean, var):

        # sams = np.random.normal(mu, sd, self.samples_per_stage)

        # no need to average over zeroes
        if T == 1:
            return 0

        # mu = self.gp.GPMean(measurements=x.history.measurements, weights=new_st.weights)

        # mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        # sd = new_st.variance

        number_of_samples = GetNumberOfSamples(self.H, T)
        # sams = np.random.multivariate_normal(mean, var, self.samples_per_stage)
        sams = np.random.multivariate_normal(mean, var, number_of_samples)

        rrr = [self.ComputeNewVRandom(T - 1, self.TransitionH(x, sam))[0] for sam in sams]
        avg = np.mean(rrr)

        return avg
    """

    def AnytimeAlgorithm(self, epsilon, x_0, H, max_nodes=10 ** 15):
        print "Preprocessing weight spaces..."

        # by default physical state length is self.batch_size
        # but for the first step it is equal to 1, since it is just agent's position

        # x_0.history.locations includes current location

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        root_ss = SemiState(physical_state, past_locations)

        # root_ss = SemiState(x_0.physical_state, x_0.history.locations[: -self.batch_size, :])

        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)
        self.PreprocessLipchitz(root_node)

        # st, new_epsilon, l, nodes_expanded=self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,epsilon)
        lamb = epsilon
        print "lambda is " + str(lamb)
        # node d_0, where we have actions
        root_action_node = MCTSActionNode(x_0, root_node, self, lamb)
        print "MCTS max nodes:", max_nodes, "Skeletal Expansion"
        # Expand tree
        total_nodes_expanded = root_action_node.SkeletalExpand()
        print "Performing search..."

        counter = 0
        # TODO: Set a proper termination condition
        # whilre resources permit
        while not root_action_node.saturated and total_nodes_expanded < max_nodes:
            print counter
            lower, upper, num_nodes_expanded = self.ConstructTree(root_action_node, root_node, H, lamb)
            total_nodes_expanded += num_nodes_expanded
            counter += 1
            if counter > 1000:
                break
        # TODO: Set action selection scheme
        # Current: Selection based on the action with the highest average bound
        # bestavg = -float('inf')
        # for a, cc in root_action_node.BoundsChildren.iteritems():
        # 	print a, cc
        # 	avg = (cc[0] + cc[1])/2
        # 	if bestavg < avg:
        # 		best_a = a
        # 		bestavg = avg



        # Select according to maximum lower bound node
        best_lower = -float('inf')
        for a, cc in root_action_node.BoundsChildren.iteritems():
            print a, cc
            if best_lower < cc[0]:
                best_a = a
                best_lower = cc[0]

        if math.isinf(best_lower):
            raise Exception("Anytime for " + str(H) + " could not move from  location " + str(x_0.physical_state))

        # bestval, best_a = self.MCTSTraverseBest(root_action_node)
        print best_lower, np.asarray(best_a)

        # Vreal, Areal, _ = self.Algorithm1(epsilon, gamma, x_0, H)
        # print Vreal, Areal

        # assert abs(Vreal-bestval) <= 0.001

        print "Total nodes expanded %d" % total_nodes_expanded
        return root_action_node.BoundsChildren[best_a], np.asarray(best_a), total_nodes_expanded

    # TODO unused
    def MCTSTraverseBest(self, action_node):
        """
        """

        if not action_node.ChanceChildren: return 0, None

        best_a = None
        best_a_val = -float('inf')
        # all  nodes d_t + s
        for a, cc in action_node.ChanceChildren.iteritems():
            # next level d_{t+1}
            v = [None] * len(cc.ActionChildren)
            for i in xrange(len(v)):
                if cc.ActionChildren[i] == None: continue
                # some of the nodes are empty
                v[i], _ = self.MCTSTraverseBest(cc.ActionChildren[i])

            # nearest neighbour
            left = [None] * len(cc.ActionChildren)
            right = [None] * len(cc.ActionChildren)

            curdist = -999999999999999999999999999999
            curval = float('inf')
            for i in xrange(len(v)):
                if not v[i] == None:
                    # Z value
                    curdist = cc.ObservationValue[i]
                    curval = v[i]
                left[i] = (curval, cc.ObservationValue[i] - curdist)

            curdist = 999999999999999999999999999999
            curval = float('inf')
            for i in reversed(xrange(len(v))):
                if not v[i] == None:
                    curdist = cc.ObservationValue[i]
                    curval = v[i]
                right[i] = (curval, curdist - cc.ObservationValue[i])

            # todo How do we get a nearest neighbour?
            # Set to nearest neighbour if none
            for i in xrange(len(v)):
                if not v[i] == None: continue
                v[i] = right[i][0]
                if left[i][1] < right[i][1]:
                    v[i] = left[i][0]

            for i in xrange(len(v)):
                # Add in sampled reward
                v[i] += self.reward_sampled(cc.ObservationValue[i])
                v[i] *= cc.IntervalWeights[i]

            sumval = sum(v) + self.reward_analytical(cc.mu, cc.semi_tree.variance)
            if sumval > best_a_val:
                best_a_val = sumval
                best_a = a

        return best_a_val, best_a

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
        self.PreprocessLipchitz(root_node)

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

    # todo need fix?
    def BuildTree(self, node, H, isRoot=False):
        """
        Builds the preprocessing (semi) tree recursively
        """
        if not isRoot:
            node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return

        # add upon crash
        """
        new_history_locations = cur_physical_state if node.ss.locations is None else np.append(node.ss.locations,
                                                                                               cur_physical_state, 0)
        """
        cur_physical_state = node.ss.physical_state
        new_locations = np.append(node.ss.locations, np.atleast_2d(cur_physical_state), 0)
        cholesky = self.gp.Cholesky(new_locations)

        # Add in new children for each valid action
        # this is for simulated
        # valid_actions = self.GetValidActionSet(node.ss.physical_state)
        # new_physical_states = [self.PhysicalTransition(cur_physical_state, a) for a in valid_actions]

        # this is for real-world
        current_location = cur_physical_state[-1, :]
        new_physical_states = self.model.GenerateRoadMacroActions(current_location, self.batch_size)

        for new_physical_state in new_physical_states:
            # Get new semi state

            # new_physical_state = self.PhysicalTransition(cur_physical_state, a)
            new_ss = SemiState(new_physical_state, new_locations)

            # Build child subtree
            new_st = SemiTree(new_ss, chol=cholesky)
            node.AddChild(new_physical_state, new_st)
            new_st.ComputeWeightsAndVariance(self.gp)
            self.BuildTree(new_st, H - 1)

    def GetValidActionSet(self, physical_state):
        return [a for a in self.macroaction_set if self.IsValidMacroAction(physical_state, a)]

    def IsValidMacroAction(self, physical_state, a):
        # TODO: ensure scalability to multiple dimensions
        # TODO: ensure epsilon comparison for floating point comparisons (currently comparing directly like a noob)

        # Physical state is a macro-action (batch)

        # both should be equal to 2, since the points are 2-d.
        # the first dimension is the length of state. should be equal to batch size
        #  but can't compare because of the first step
        assert physical_state.shape[1] == a.shape[1]
        new_state = PhysicalTransition(physical_state, a)
        assert new_state.shape == a.shape
        # print new_state
        ndims = 2
        eps = 0.001
        # a.shape[0] is batch_size
        for i in range(a.shape[0]):
            current_agent_postion = new_state[i, :]
            for dim in xrange(ndims):
                if current_agent_postion[dim] < self.grid_domain[dim][0] or current_agent_postion[dim] >= \
                        self.grid_domain[dim][1]:
                    return False

        # Check for obstacles
        """
        if self.bad_places:
            for j in xrange(len(self.bad_places)):
                if abs(current_agent_postion[0] - (self.bad_places[j])[0]) < eps and abs(
                                current_agent_postion[1] - (self.bad_places[j])[1]) < eps:
                    return False
        """
        # print "state is " + str(new_state)
        return True

    def PreprocessLipchitz(self, node):
        """
        Obtain Lipchitz vector and Lipchitz constant for each node.
        @param node - root node of the semi-tree (assumed to be already constructed)
        """

        # Base case
        if len(node.children) == 0:
            # now L_upper is a scalar
            # node.L_upper = np.zeros((nl + 1, 1))
            lip = 0

        else:
            # Recursive case
            # vmax = np.zeros((nl + 1, 1))
            vmax = 0
            for _, c in node.children.iteritems():
                self.PreprocessLipchitz(c)
                # do we need to count all the values L_upper if use only one?
                # av = (c.L_upper[0:nl + 1] + ((c.L_upper[-1] + self.l1 + self.l2(math.sqrt(c.variance))) * (c.weights.T)))
                alpha = np.linalg.norm(c.weights, ord='fro')
                av = c.lipchitz * math.sqrt(1 + alpha ** 2) + alpha * math.sqrt(self.batch_size)
                vmax = np.maximum(vmax, av)
                lip = vmax

        # node.lipchitz = node.L_upper[-1]
        node.lipchitz = lip

    """
    def EstimateV(self, T, l, gamma, x, st):
        @return vBest - approximate value function computed
        @return aBest - action at the root for the policy defined by alg1
        @param st - root of the semi-tree to be used


        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[ToTuple(a)]

            # Reward is just the mean added to a multiple of the variance at that point
            # todo fix
            mean = self.gp.GPMean(locations=x_next.history.locations, measurements=x_next.history.measurements,
                                  current_location=x_next.physical_state)
            var = new_st.variance
            r = self.reward_analytical(mean, var)

            # Future reward
            f = self.Q_det(T, l, gamma, x_next, new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def Q_det(self, T, l, gamma, x, new_st):
        Approximates the integration step derived from alg1
        @param new_st - semi-tree at this stage
        @return - approximate value of the integral/expectation


        # if T > 3: print T
        # Initialize variables
        kxi = new_st.lipchitz
        # todo fix to batch
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = math.sqrt(new_st.variance)
        k = new_st.k
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
            v, _ = self.EstimateV(T - 1, l, gamma, self.TransitionH(x, zPoints), new_st)
            # v += self.reward_sampled(zPoints)

            # Recursively compute values
            vAccum += v * (norm.cdf(x=zUpper, loc=mu, scale=sd) - norm.cdf(x=zLower, loc=mu, scale=sd))
            testsum += (norm.cdf(x=zUpper, loc=mu, scale=sd) - norm.cdf(x=zLower, loc=mu, scale=sd))

        # Weight values
        rightLimit = mu + k * sd
        leftLimit = mu - k * sd
        vRightTailVal, _ = self.EstimateV(T - 1, l, gamma, self.TransitionH(x, rightLimit), new_st)
        # vRightTailVal += self.reward_sampled(rightLimit)
        if num_partitions == 2:
            vLeftTailVal = vRightTailVal  # Quick hack for cases where the algo only requires us to us MLE (don't need to repeat measurement at mean of gaussian pdf)
        else:
            vLeftTailVal, _ = self.EstimateV(T - 1, l, gamma, self.TransitionH(x, leftLimit), new_st)  # usual case
            # vLeftTailVal += self.reward_sampled(leftLimit)
        vAccum += vRightTailVal * (1 - norm.cdf(x=rightLimit, loc=mu, scale=sd)) + \
                  vLeftTailVal * norm.cdf(x=leftLimit, loc=mu, scale=sd)
        testsum += (1 - norm.cdf(x=rightLimit, loc=mu, scale=sd)) + norm.cdf(x=leftLimit, loc=mu, scale=sd)
        assert abs(testsum - 1.0) < 0.0001, "Area != 1, %f instead" % testsum

        return vAccum

        # return vAccum
    """
    def ConstructTree(self, action_node, st, T, l):

        if T == 0: return 0, 0, 0
        assert not action_node.saturated, "Exploring saturated action node"

        # Select action that has the greatest upper bound (TODO: make sure there are still leaves in that branch)
        highest_upper = -float('inf')
        best_a = None
        # choose the best child so far
        for a, bounds in action_node.BoundsChildren.iteritems():
            if action_node.ChanceChildren[a].saturated: continue
            if highest_upper < bounds[1]: best_a = a
            highest_upper = max(highest_upper, bounds[1])

        new_semi_tree = st.children[best_a]

        # Select observation that has the greatest WEIGHTED error
        obs_node = action_node.ChanceChildren[best_a]
        highest_variance = -0.5
        most_uncertain_node_index = None
        for i in xrange(obs_node.num_samples):
            if not (obs_node.ActionChildren[i] is None) and obs_node.ActionChildren[i].saturated:
                continue
            current_variance = obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0]
            if current_variance > highest_variance:
                most_uncertain_node_index = i
                highest_variance = current_variance

        i = most_uncertain_node_index
        # If observation is leaf, then we expand:
        if obs_node.ActionChildren[i] is None:

            new_action_node = MCTSActionNode(TransitionH(obs_node.augmented_state, obs_node.ObservationValue[i]),
                                             new_semi_tree, self, l)
            obs_node.ActionChildren[i] = new_action_node

            num_nodes_expanded = new_action_node.SkeletalExpand()
            # Update upper and lower bounds on this observation node
            lower, upper = new_action_node.Eval()
            obs_node.BoundsChildren[i] = (
                max(obs_node.BoundsChildren[i][0], lower), min(obs_node.BoundsChildren[i][1], upper))

        else:  # Observation has already been made, expand further

            lower, upper, num_nodes_expanded = self.ConstructTree(obs_node.ActionChildren[i], new_semi_tree, T - 1, l)
            obs_node.BoundsChildren[i] = (
                max(lower, obs_node.BoundsChildren[i][0]), min(upper, obs_node.BoundsChildren[i][1]))

        obs_node.UpdateChildrenBounds(i)
        lower, upper = obs_node.Eval()
        assert (lower <= upper)
        action_node.BoundsChildren[best_a] = (
            max(action_node.BoundsChildren[best_a][0], lower), min(action_node.BoundsChildren[best_a][1], upper))

        if obs_node.ActionChildren[i].saturated:
            obs_node.numchild_unsaturated -= 1
            if obs_node.numchild_unsaturated == 0:
                obs_node.saturated = True

        # action_node.DetermineDominance()
        action_node.DetermineSaturation()

        return action_node.Eval() + (num_nodes_expanded,)

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
    # new macroaction
    new_augmented_state.physical_state = PhysicalTransition(new_augmented_state.physical_state, action)
    return new_augmented_state


def TransitionH(augmented_state, measurements):
    """
        @return - copy of augmented state with history updated
        """
    new_augmented_state = copy.deepcopy(augmented_state)
    # add new batch and measurements
    new_augmented_state.history.append(new_augmented_state.physical_state, measurements)
    return new_augmented_state


def PhysicalTransition(physical_state, macroaction):
    """
        @param - physical_state: numpy array with same size as action
        @return - new physical state after taking the action
        :param macroaction:
        """

    current_location = physical_state[-1, :]
    batch_size = macroaction.shape[0]

    repeated_location = np.asarray([current_location for i in range(batch_size)])
    # repeated_location = np.tile(current_location, batch_size)

    assert repeated_location.shape == macroaction.shape
    # new physical state is a batch starting from the current location (the last element of batch)
    new_physical_state = np.add(repeated_location, macroaction)

    # check that it is 2d
    assert new_physical_state.ndim == 2
    return new_physical_state


# updated
# just state and history
class AugmentedState:
    def __init__(self, physical_state, initial_history):
        """
        Initialize augmented state with initial position and history
        """
        # 2D array
        self.physical_state = physical_state
        # 2D array
        self.history = initial_history

    def to_str(self):
        return \
            "Physical State\n" + \
            str(self.physical_state) + "\n" + \
            "Locations\n" + \
            str(self.history.locations) + "\n" + \
            "Measurements\n" + \
            str(self.history.measurements)


# UTIL
# converts ndarray to tuple
# can't pass ndarray as a key for dict
def ToTuple(arr):
    return tuple(map(tuple, arr))


# updated
class SemiTree:
    def __init__(self, semi_state, chol=None):
        self.ss = semi_state
        self.children = dict()
        self.weights = None  # Weight space vector
        self.variance = None  # Precomputed posterior variance
        self.cholesky = chol

    def AddChild(self, action, semi_tree):
        key = ToTuple(action)
        self.children[key] = semi_tree

        # todo refact

    def ComputeWeightsAndVariance(self, gp):
        """ Compute the weights for this semi_state ONLY"""
        self.weights, self.variance = gp.GetBatchWeightsAndVariance(self.ss.locations, self.ss.physical_state,
                                                                    self.cholesky)
        # self.variance = gp.GPVariance(self.ss.locations, self.ss.physical_state, self.cholesky)


# updated
class SemiState:
    """ State which only contains locations visited and its current location
    """

    def __init__(self, physical_state, locations):
        self.physical_state = physical_state
        # hsitory doesn't include current batch
        # it is more convenient since we do not know mean and var in current batch
        # and locations is history for predicting it
        self.locations = locations


class MCTSActionNode:
    """
    """
    # d_t
    mini_epsilon = 10 ** -8

    def __init__(self, augmented_state, semi_tree, treeplan, l):
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan

        # number of samples per stage at every child ObservationNode
        self.number_of_samples = self.treeplan.samples_per_stage

        self.lamb = l

        # is full?
        self.saturated = False
        # d_t + s_{t]
        self.ChanceChildren = dict()
        # Q_lower and Q_upper for each child
        self.BoundsChildren = dict()
        self.max_upper = -float('inf')
        self.max_lower = -float('inf')

    def Eval(self):
        """
        Evaluate upper and lower bounds of this action node
        V_lower and V_upper
        """
        # if no children, then we are certain, and zero value
        if len(self.BoundsChildren) == 0:
            return -MCTSActionNode.mini_epsilon, MCTSActionNode.mini_epsilon

        # max_upper = -float('inf')
        # max_lower = -float('inf')
        # get max upper and lower bound
        # V_lower and V_upper
        for _, b in self.BoundsChildren.iteritems():
            self.max_upper = max(b[1], self.max_upper)
            self.max_lower = max(b[0], self.max_lower)

        # todo refact
        # self.max_upper = max_upper
        # self.max_lower = max_lower

        return self.max_lower, self.max_upper

    def SkeletalExpand(self):
        """ Builds observation nodes for every action
        """

        # generate all children d_t + s_{t+1}
        num_nodes_expanded = 1
        for a, semi_child in self.semi_tree.children.iteritems():
            # TODO bad design
            next_physical_state = np.asarray(a)
            fake_action  = np.zeros(next_physical_state.shape)
            next_augmented_state = TransitionP(self.augmented_state, fake_action)
            next_augmented_state.physical_state = next_physical_state

            c = MCTSObservationNode(next_augmented_state, semi_child, self.treeplan,
                                    self.lamb,
                                    self.number_of_samples)
            num_nodes_expanded += c.SkeletalExpand()
            self.ChanceChildren[a] = c
            self.BoundsChildren[a] = c.Eval()

        # self.DetermineDominance()
        self.DetermineSaturation()
        return num_nodes_expanded

    # not used
    def DetermineDominance(self):

        dominated = True

        # Get action with the highest lower bound (may not be the best action per se)
        highest_lower = -float('inf')
        for a, cc in self.ChanceChildren.iteritems():
            if self.BoundsChildren[a][0] >= highest_lower:
                highest_lower = self.BoundsChildren[a][0]
                best_a = a

        # Check dominance
        for a, cc in self.ChanceChildren.iteritems():
            if a == best_a: continue
            if self.BoundsChildren[a][1] < highest_lower:
                # pass
                # if not cc.saturated: print "Action %s cutoff in favour of %s" % (a, best_a)
                cc.saturated = True  # saturate all nodes which are dominated

    def DetermineSaturation(self):
        """ Action node is saturated when
        Everything underneath it is saturated (or dominated)
        """
        allSat = True
        for a, cc in self.ChanceChildren.iteritems():
            if not cc.saturated: allSat = False

        # todo check if works initially was like this
        # self.saturated = allSat
        self.saturated = allSat


class MCTSObservationNode:
    def __init__(self, augmented_state, semi_tree, treeplan, l, number_of_samples):
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan
        self.lamb = l

        self.num_samples = number_of_samples
        """
        # todo need to change to stochastic samples
        # Number of partitions INCLUDING tails
        if self.semi_tree.n == 0:
            self.num_samples = 1  # MLE case
        else:
            self.num_samples = self.semi_tree.n + 2
        """
        self.saturated = False

        self.numchild_unsaturated = self.num_samples

        self.mu = self.treeplan.gp.GPMean(measurements=augmented_state.history.measurements, weights=semi_tree.weights)

        # self.mu = self.treeplan.gp.GPMean(augmented_state.history.locations, augmented_state.history.measurements,
        #                                  augmented_state.physical_state, weights=semi_tree.weights)

        # Pointer too children action selection nodes. "None" = this observation has not been expanded.
        self.ActionChildren = [None] * self.num_samples
        # Array of (lower, upper) tuple. Includes bounds which are due to Lipschitz constraints.
        self.BoundsChildren = [(-float('inf'), float('inf'))] * self.num_samples
        # todo check if we need it
        # Range of observations for this partition
        # self.ObservationBounds = [None] * self.num_samples
        # todo change into stochastic sample values
        # Value of observation that we take from this partition
        # self.ObservationValue = [None] * self.num_samples

        mu = self.mu
        # sd = math.sqrt(semi_tree.variance)
        samples = np.random.multivariate_normal(mu, semi_tree.variance, self.num_samples)
        # samples = np.random.normal(mu, sd, self.num_samples)

        # cannot sort
        # self.ObservationValue = np.sort(samples, axis=None)
        self.ObservationValue = samples
        """
        # todo remove
        # Weight of each interval
        # self.IntervalWeights = [None] * self.num_samples

        #################################################
        # Compute partition information when NOT mle
        #################################################
        if self.num_samples > 1:

            # Initialize variables
            # kxi = semi_tree.lipchitz

            k = semi_tree.k

            if semi_tree.n > 0: width = 2.0 * k * sd / semi_tree.n
            for i in xrange(2, self.num_samples):
                # Compute boundary points
                zLower = mu - sd * k + (i - 2) * width
                zUpper = mu - sd * k + (i - 1) * width
                self.ObservationBounds[i - 1] = (zLower, zUpper)

                # Compute evaluation points
                self.ObservationValue[i - 1] = 0.5 * (zLower + zUpper)

                # Compute weights
                self.IntervalWeights[i - 1] = norm.cdf(x=zUpper, loc=mu, scale=sd) - norm.cdf(x=zLower, loc=mu,
                                                                                              scale=sd)

            # Values for extremes
            rightLimit = mu + k * sd
            leftLimit = mu - k * sd
            self.ObservationBounds[0] = (-float('inf'), leftLimit)
            self.ObservationBounds[-1] = (rightLimit, float('inf'))
            self.ObservationValue[0] = leftLimit
            self.ObservationValue[-1] = rightLimit
            self.IntervalWeights[0] = norm.cdf(x=leftLimit, loc=mu, scale=sd)
            self.IntervalWeights[-1] = 1 - norm.cdf(x=rightLimit, loc=mu, scale=sd)

            assert abs(sum(self.IntervalWeights) - 1) < 0.0001, "Area != 1, %f instead\n With number: %s " % (
                sum(self.IntervalWeights), str(self.IntervalWeights))

        else:
            #################################################
            # Compute partition information when using mle
            #################################################

            self.ObservationBounds[0] = (-float('inf'), float('inf'))
            self.ObservationValue[0] = self.mu
            self.IntervalWeights[0] = 1.0
        """

    def Eval(self):
        """
        Evaluate upper and lower bounds of this chance node (weighted)
       they are Q_lower and Q_upper
        """

        # lower = 0.0
        # upper = 0.0
        # the same as num_samples
        number_of_children = len(self.BoundsChildren)

        lower = sum([childBound[0] for childBound in self.BoundsChildren]) / number_of_children
        upper = sum([childBound[1] for childBound in self.BoundsChildren]) / number_of_children

        """
        for i in xrange(len(self.BoundsChildren)):
            lower += self.BoundsChildren[i][0] * self.IntervalWeights[i]
            upper += self.BoundsChildren[i][1] * self.IntervalWeights[i]
        """
        # Update reward
        # lower += self.mu - self.semi_tree.true_error
        # upper += self.mu + self.semi_tree.true_error
        r = self.treeplan.reward_analytical(self.mu, self.semi_tree.variance)

        lower += r - self.lamb
        upper += r + self.lamb

        assert (lower <= upper), "Lower > Upper!, %s, %s" % (lower, upper)

        return lower, upper

    def UpdateChildrenBounds(self, index_updated):
        """ Update bounds of OTHER children while taking into account lipschitz constraints
        @param index_updated: index of child whose bound was just updated
        """

        lip = self.semi_tree.lipchitz

        assert self.BoundsChildren[index_updated][0] <= self.BoundsChildren[index_updated][1], "%s, %s" % (
            self.BoundsChildren[index_updated][0], self.BoundsChildren[index_updated][1])

        for i in range(len(self.ActionChildren)):
            # is it efficient? or better remove from iteration list?
            if i == index_updated:
                continue
            # line 20 of algorithm in draft
            b = np.linalg.norm(self.ObservationValue[i] - self.ObservationValue[index_updated]) * lip
            testLower = self.BoundsChildren[index_updated][0] - b
            testUpper = self.BoundsChildren[index_updated][1] + b
            # print self.BoundsChildren[i], testLower, testUpper
            if self.BoundsChildren[i][0] < testLower:
                self.BoundsChildren[i] = (testLower, self.BoundsChildren[i][1])

            if self.BoundsChildren[i][1] > testUpper:
                self.BoundsChildren[i] = (self.BoundsChildren[i][0], testUpper)

            assert (
                self.BoundsChildren[i][0] <= self.BoundsChildren[i][
                    1]), "lower bound greater than upper bound %f, %f" % (
                self.BoundsChildren[i][0], self.BoundsChildren[i][1])
        """
        # todo no more left and right!
        # Intervals lying to the left of just updated interval
        for i in reversed(xrange(index_updated)):
            change = False
            testLower = self.BoundsChildren[i + 1][0] - lip * (self.ObservationValue[i + 1] - self.ObservationValue[i])
            testUpper = self.BoundsChildren[i + 1][1] + lip * (self.ObservationValue[i + 1] - self.ObservationValue[i])
            # print self.BoundsChildren[i], testLower, testUpper
            if self.BoundsChildren[i][0] < testLower:
                change = True
                self.BoundsChildren[i] = (testLower, self.BoundsChildren[i][1])

            if self.BoundsChildren[i][1] > testUpper:
                change = True
                self.BoundsChildren[i] = (self.BoundsChildren[i][0], testUpper)

            assert (
                self.BoundsChildren[i][0] <= self.BoundsChildren[i][
                    1]), "lower bound greater than upper bound %f, %f" % (
                self.BoundsChildren[i][0], self.BoundsChildren[i][1])

            if not change == True:
                break

        # Intervals lying to the right of just updated interval
        for i in xrange(index_updated + 1, len(self.ActionChildren)):
            change = False
            testLower = self.BoundsChildren[i - 1][0] - lip * (self.ObservationValue[i] - self.ObservationValue[i - 1])
            testUpper = self.BoundsChildren[i - 1][1] + lip * (self.ObservationValue[i] - self.ObservationValue[i - 1])
            if self.BoundsChildren[i][0] < testLower:
                change = True
                self.BoundsChildren[i] = (testLower, self.BoundsChildren[i][1])

            if self.BoundsChildren[i][1] > testUpper:
                change = True
                self.BoundsChildren[i] = (self.BoundsChildren[i][0], testUpper)

            assert (
                self.BoundsChildren[i][0] <= self.BoundsChildren[i][
                    1]), "lower bound greater than upper bound %f, %f" % (
                self.BoundsChildren[i][0], self.BoundsChildren[i][1])

            if not change == True:
                break
        """

    def SkeletalExpand(self):
        """ Expand only using observations at the edges
        """

        num_nodes_expanded = 0
        # # Note Special case of MLE where only one expansion is done
        # num_nodes_expanded += self.SkeletalExpandHere(0)
        # self.UpdateChildrenBounds(0)

        # if self.num_partitions > 1:
        # 	num_nodes_expanded += self.SkeletalExpandHere(self.num_partitions-1)
        # 	self.UpdateChildrenBounds(self.num_partitions-1)

        # choose the center node
        # todo change coz ugly
        list_observations = self.ObservationValue.tolist()
        distances = [np.linalg.norm(observation - self.mu) for observation in list_observations]
        target = -1.0
        current_min = float('inf')
        for i in range(len(distances)):
            if distances[i] < current_min:
                current_min = distances[i]
                target = i

        assert target >= 0
        # target = int(math.floor(self.num_samples / 2))
        num_nodes_expanded += self.SkeletalExpandHere(target)
        self.UpdateChildrenBounds(target)
        return num_nodes_expanded

    def SkeletalExpandHere(self, index_to_expand):
        """ Expand given node at a particular index
        """
        num_nodes_expanded = 0
        assert self.ActionChildren[index_to_expand] == None, "Node already expanded"
        # uses obervation value
        self.ActionChildren[index_to_expand] = MCTSActionNode(
            TransitionH(self.augmented_state, self.ObservationValue[index_to_expand]), self.semi_tree, self.treeplan,
            self.lamb)
        num_nodes_expanded += self.ActionChildren[index_to_expand].SkeletalExpand()
        lower, upper = self.ActionChildren[index_to_expand].Eval()
        assert lower <= upper
        # print lower, upper, self.BoundsChildren[index_to_expand]
        self.BoundsChildren[index_to_expand] = (
            # tighten the bounds
            max(self.BoundsChildren[index_to_expand][0], lower), min(self.BoundsChildren[index_to_expand][1], upper))
        # print self.BoundsChildren[index_to_expand]
        assert self.BoundsChildren[index_to_expand][0] <= self.BoundsChildren[index_to_expand][1]
        self.UpdateChildrenBounds(index_to_expand)

        if self.ActionChildren[index_to_expand].saturated:
            self.numchild_unsaturated -= 1
            if self.numchild_unsaturated == 0: self.saturated = True

        return num_nodes_expanded


# updated
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
        assert new_locations.ndim == 2
        assert self.locations.ndim == 2
        self.locations = np.append(self.locations, new_locations, axis=0)
        # 1D array

        assert self.measurements.ndim == 1
        assert new_measurements.ndim == 1

        self.measurements = np.append(self.measurements, new_measurements)


if __name__ == "__main__":
    pass
