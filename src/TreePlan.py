"""

Implements Algorithm I in my CA report

- iterative Deepening
- epsilon optimal

"""

import copy
import math

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm

from GaussianProcess import GaussianProcess
from GaussianProcess import SquareExponential
from Vis2d import Vis2d
from mutil import mutil
from src.MacroActionGenerator import GenerateSimpleMacroactions


class TreePlan:
    """
    TODO: allow for more flexible initialization
    def __init__(self, states, actions, transition, reward, GP):
        pass
    """

    static_mathutil = mutil()
    static_mathutil.Init(200)

    # static_mathutil.Init(25000)

    def __init__(self, batch_size, grid_domain, grid_gap, num_samples, gaussian_process, macroaction_set=None,
                 max_nodes=None,
                 sd_bonus=0.0, bad_places=None):
        """
        - Gradularity given by grid_gap
        - Squared exponential covariance function
        - Characteristic length scale the same for both directions
        """

        self.batch_size = batch_size
        # Preset constants
        self.INF = 10 ** 15

        # Number of observations/samples generated for every node
        self.samples_per_stage = num_samples

        # Problem parameters
        self.grid_gap = grid_gap
        self.macroaction_set = macroaction_set
        if macroaction_set is None:
            self.macroaction_set = GenerateSimpleMacroactions(self.batch_size, self.grid_gap)

        self.grid_domain = grid_domain
        self.gp = gaussian_process
        self.max_nodes = self.INF if max_nodes is None else max_nodes
        self.sd_bonus = sd_bonus

        # Obstacles
        self.bad_places = bad_places

        # Precomputed algo stuff
        # todo do we need this?
        self.mathutil = TreePlan.static_mathutil

        self.l1 = 0
        # todo chenge
        self.l2 = lambda sigma: 1

        self.reward_analytical = lambda mu, sigma: self.AcquizitionFunction(mu, sigma)
        self.reward_sampled = lambda f: 0


    # heuristic
    # we use batch UCB version from Erik
    # todo check that we do not add noise twice
    def AcquizitionFunction(self, mu, sigma):
        exploration_matrix = np.identity(sigma.shape[0]) * (self.gp.noise_variance) ** (2) + sigma

        return np.sum(mu) + self.beta * math.log(np.linalg.det(exploration_matrix))

    def Algorithm1(self, epsilon, gamma, x_0, H):
        """
                @param x_0 - augmented state
                @return approximately optimal value, answer, and number of node expansions
                """

        # Obtain lambda
        # l = epsilon / (gamma * H * (H+1))
        # l = epsilon / sum([gamma ** i for i in xrange(1, H+1)])

        print "Preprocessing weight spaces..."
        # Obtain Lipchitz lookup tree
        st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[: -self.batch_size, :], H,
                                                             epsilon)

        print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.EstimateV(H, l, gamma, x_0, st)

        return Vapprox, Aapprox, nodes_expanded

        return Vapprox, Aapprox, nodes_expanded

    def MLE(self, x_0, H):
        """
                @param x_0 - augmented state
                @return approximately optimal value, answer, and number of node expansions
        """
        root_ss = SemiState(x_0.physical_state, x_0.history.locations[: -self.batch_size, :])
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        # st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,epsilon)

        # print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.ComputeVMLE(H, x_0, root_node)

        return Vapprox, Aapprox, -1

    def ComputeVMLE(self, T, x, st):

        """
                @return vBest - approximate value function computed
                @return aBest - action at the root for the policy defined by alg1
                @param st - root of the semi-tree to be used
                """

        valid_actions = self.GetValidActionSet(x.physical_state)
        # not needed
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[ToTuple(a)]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.ComputeQMLE(T, x_next, new_st) + r

            if f > vBest:
                aBest = a
                vBest = f

        return vBest, aBest

    def ComputeQMLE(self, T, x, new_st):
        # no need to average over zeroes
        if T == 1:
            return 0
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)
        return self.ComputeVRandom(T - 1, self.TransitionH(x, mu), new_st)[0]

    def StochasticFull(self, x_0, H):
        """
                @param x_0 - augmented state
                @return approximately optimal value, answer, and number of node expansions
        """
        root_ss = SemiState(x_0.physical_state, x_0.history.locations[: -self.batch_size, :])
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        # st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,epsilon)

        # print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.ComputeVRandom(H, x_0, root_node)

        return Vapprox, Aapprox, -1

    def ComputeVRandom(self, T, x, st):

        """
                @return vBest - approximate value function computed
                @return aBest - action at the root for the policy defined by alg1
                @param st - root of the semi-tree to be used
                """

        valid_actions = self.GetValidActionSet(x.physical_state)
        # not needed
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[ToTuple(a)]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, var)

            # Future reward
            f = self.ComputeQRandom(T, x_next, new_st) + r

            if f > vBest:
                aBest = a
                vBest = f

        return vBest, aBest

    def ComputeQRandom(self, T, x, new_st):

        #sams = np.random.normal(mu, sd, self.samples_per_stage)

        # no need to average over zeroes
        if T == 1:
            return 0

        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = new_st.variance

        sams = np.random.multivariate_normal(mu, sd, self.samples_per_stage)

        rrr = [self.ComputeVRandom(T - 1, self.TransitionH(x, sam),
                                   new_st)[0] for sam in sams]
        avg = np.mean(rrr)

        return avg

    def FindMLEError(self, st):

        if not st.children: return 0;

        max_err = 0
        for a, semi_child in st.children.iteritems():
            new_err = math.sqrt(2 / math.pi) * (semi_child.lipchitz + self.l1) * math.sqrt(semi_child.variance)
            rec_err = self.FindMLEError(semi_child)
            max_err = max(max_err, new_err + rec_err)

        return max_err

    def AnytimeAlgorithm(self, epsilon, x_0, H, max_nodes=10 ** 15):
        print "Preprocessing weight spaces..."

        root_ss = SemiState(x_0.physical_state, x_0.history.locations[: -self.batch_size, :])
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)
        self.PreprocessLipchitz(root_node, isRoot=True)

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

        # bestval, best_a = self.MCTSTraverseBest(root_action_node)
        print best_lower, best_a

        # Vreal, Areal, _ = self.Algorithm1(epsilon, gamma, x_0, H)
        # print Vreal, Areal

        # assert abs(Vreal-bestval) <= 0.001

        print "Total nodes expanded %d" % total_nodes_expanded
        return root_action_node.BoundsChildren[best_a], best_a, total_nodes_expanded

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

            sumval = sum(v) + self.reward_analytical(cc.mu, math.sqrt(cc.semi_tree.variance))
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

    # todo need fix?
    def BuildTree(self, node, H, isRoot=False):
        """
        Builds the preprocessing (semi) tree recursively
        """

        if not isRoot: node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return

        # add upon crash
        """
        new_history_locations = cur_physical_state if node.ss.locations is None else np.append(node.ss.locations,
                                                                                               cur_physical_state, 0)
        """

        # Add in new children for each valid action
        valid_actions = self.GetValidActionSet(node.ss.physical_state)
        for a in valid_actions:
            # Get new semi state
            cur_physical_state = node.ss.physical_state
            new_physical_state = self.PhysicalTransition(cur_physical_state, a)
            new_locations = np.append(node.ss.locations, np.atleast_2d(cur_physical_state), 0)
            new_ss = SemiState(new_physical_state, new_locations)

            # Build child subtree
            new_st = SemiTree(new_ss)
            node.AddChild(a, new_st)
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
        # print new_state
        ndims = 2
        eps = 0.001
        for i in range(a.shape[0]):
            current_agent_postion = new_state[i, :]
            for dim in xrange(ndims):
                if current_agent_postion[dim] < self.grid_domain[dim][0] or current_agent_postion[dim] >= \
                        self.grid_domain[dim][1]: return False

        # Check for obstacles
        if self.bad_places:
            for j in xrange(len(self.bad_places)):
                if abs(current_agent_postion[0] - (self.bad_places[j])[0]) < eps and abs(
                                current_agent_postion[1] - (self.bad_places[j])[1]) < eps:
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

    def EstimateV(self, T, l, gamma, x, st):
        """
        @return vBest - approximate value function computed
        @return aBest - action at the root for the policy defined by alg1
        @param st - root of the semi-tree to be used
        """

        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0, valid_actions[0]

        vBest = -self.INF
        aBest = valid_actions[0]
        for a in valid_actions:

            x_next = self.TransitionP(x, a)

            # go down the semitree node
            new_st = st.children[ToTuple(a)]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.Q_det(T, l, gamma, x_next, new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def Q_det(self, T, l, gamma, x, new_st):
        """
        Approximates the integration step derived from alg1
        @param new_st - semi-tree at this stage
        @return - approximate value of the integral/expectation
        """

        # if T > 3: print T
        # Initialize variables
        kxi = new_st.lipchitz
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
    # todo check dimensions
    current_location = physical_state[-1, :]
    batch_size = macroaction.shape[0]
    # todo fix cause very ugly
    repeated_location = np.asarray([current_location for i in range(batch_size)])
    # repeated_location = np.tile(current_location, batch_size)
    assert repeated_location.shape == macroaction.shape
    # new physical state is a batch starting from the current location (the last element of batch)
    new_physical_state = np.add(repeated_location, macroaction)

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
        chol = gp.GPCholTraining(self.ss.locations)
        cov_query = gp.GPCovQuery(self.ss.locations, self.ss.physical_state)
        self.weights = gp.GPWeights(self.ss.locations, self.ss.physical_state, chol, cov_query)
        self.variance = gp.GPVariance2(self.ss.locations, self.ss.physical_state, chol, cov_query)


# updated
class SemiState:
    """ State which only contains locations visited and its current location
    """

    # TODO locations include current state?
    def __init__(self, physical_state, locations):
        self.physical_state = physical_state
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
            # d_t + s_{t+1}
            c = MCTSObservationNode(TransitionP(self.augmented_state, a), semi_child, self.treeplan, self.lamb,
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
        self.mu = self.treeplan.gp.GPMean(augmented_state.history.locations, augmented_state.history.measurements,
                                          augmented_state.physical_state, weights=semi_tree.weights)

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
        sd = math.sqrt(semi_tree.variance)

        samples = np.random.normal(mu, sd, self.num_samples)
        self.ObservationValue = np.sort(samples, axis=None)
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
        # todo check if we need this true error? defined in get partitions
        r = self.treeplan.reward_analytical(self.mu, math.sqrt(self.semi_tree.variance))
        # todo change to lambda
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
        # todo change into minimal distance
        target = int(math.floor(self.num_samples / 2))
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
        self.locations = np.append(self.locations, new_locations, axis=0)
        # 1D array
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

        print tp.AnytimeAlgorithm(epsilon, gamma, x_0, H)

        _, a, _ = tp.Algorithm1(epsilon, gamma, x_0, H)

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
