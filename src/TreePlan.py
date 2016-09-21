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

    def Algorithm1(self, epsilon, gamma, x_0, H):
        """
        @param x_0 - augmented state
        @return approximately optimal value, answer, and number of node expansions
        """

        # Obtain lambda
        # l = epsilon / (gamma * H * (H+1))
        # l = epsilon / sum([gamma ** i for i in xrange(1, H+1)])

        #print "Preprocessing weight spaces..."
        # Obtain Lipchitz lookup tree
        st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,
                                                             epsilon)

        #print "Performing search..."
        # Get answer
        Vapprox, Aapprox = self.EstimateV(H, l, gamma, x_0, st)

        return Vapprox, Aapprox, nodes_expanded

    def RandomSampling(self, epsilon, x_0, H):
        print "gururu"

        beta = epsilon / H
        e_s = beta / 4

        # max_err = self.FindMLEError(st)
        max_err = 1.0
        lamb_d = max_err / H

        delta = min(beta / 8 / max_err, 1)
        lamb = e_s / H

        st, _, __, ___ = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H, max_err)

        print "max error", max_err
        print "delta", delta

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
            f = self.EstimateV_tilde(H, lamb_d, 1, x_next, new_st) + r  # using MLE
            frandom = self.ComputeQRandom(H, lamb, x_next, 1.0 - delta, new_st) + r

            # Correction step
            qmod = frandom
            if abs(frandom - f) > e_s + max_err:
                qmod = f

            if (qmod > vBest):
                aBest = a
                vBest = qmod

        return vBest, aBest

    def ComputeVRandom(self, T, l, x, p, st):

        valid_actions = self.GetValidActionSet(x.physical_state)
        if T == 0: return 0

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

        return vBest

    def ComputeQRandom(self, T, l, x, p, new_st):
        # print "Q: p", p
        # Initialize variables
        mu = self.gp.GPMean(x.history.locations, x.history.measurements, x.physical_state, weights=new_st.weights)

        sd = math.sqrt(new_st.variance)

        n = math.ceil(
            2 * math.log(0.5 - 0.5 * (p ** self.PEWPEW)) * ((self.l1 + new_st.lipchitz) ** 2) * new_st.variance / (
            -(l ** 2)))
        n = max(n, 1)
        if n > 1: print n

        sams = np.random.normal(mu, sd, n)

        rrr = [self.ComputeVRandom(T - 1, l, self.TransitionH(x, sam), p ** ((1 - self.PEWPEW) / n),
                                   new_st) + self.reward_sampled(sam) for sam in sams]
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

    def MCTSExpand(self, epsilon, gamma, x_0, H, max_nodes=10 ** 15):
        print "Preprocessing weight spaces..."
        st, new_epsilon, l, nodes_expanded = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H,
                                                             epsilon)

        root_action_node = MCTSActionNode(x_0, st, self, l)
        print "MCTS max nodes:", max_nodes, "Skeletal Expansion"
        total_nodes_expanded = root_action_node.SkeletalExpand()
        print "Performing search..."

        # TODO: Set a proper termination condition
        while not root_action_node.saturated and total_nodes_expanded < max_nodes:
            lower, upper, num_nodes_expanded = self.MCTSRollout(root_action_node, st, H, l)
            total_nodes_expanded += num_nodes_expanded

        # TODO: Set action selection scheme
        # Current: Selection based on the action with the highest average bound
        # bestavg = -float('inf')
        # for a, cc in root_action_node.BoundsChildren.iteritems():
        # 	print a, cc
        # 	avg = (cc[0] + cc[1])/2
        # 	if bestavg < avg:
        # 		best_a = a
        # 		bestavg = avg

        # Select according to maximum node
        # best_upper = -float('inf')
        # for a, cc in root_action_node.BoundsChildren.iteritems():
        # 	print a, cc
        # 	if best_upper < cc[1]:
        # 		best_a = a
        # 		best_upper = cc[1]

        bestval, best_a = self.MCTSTraverseBest(root_action_node)
        print bestval, best_a

        # Vreal, Areal, _ = self.Algorithm1(epsilon, gamma, x_0, H)
        # print Vreal, Areal

        # assert abs(Vreal-bestval) <= 0.001

        print "Total nodes expanded %d" % total_nodes_expanded
        return root_action_node.BoundsChildren[best_a], best_a, total_nodes_expanded

    def MCTSTraverseBest(self, action_node):
        """
        """

        if not action_node.ChanceChildren: return 0, None

        best_a = None
        best_a_val = -float('inf')
        for a, cc in action_node.ChanceChildren.iteritems():
            v = [None] * len(cc.ActionChildren)
            for i in xrange(len(v)):
                if cc.ActionChildren[i] == None: continue
                v[i], _ = self.MCTSTraverseBest(cc.ActionChildren[i])

            # nearest neighbour
            left = [None] * len(cc.ActionChildren)
            right = [None] * len(cc.ActionChildren)

            curdist = -999999999999999999999999999999
            curval = float('inf')
            for i in xrange(len(v)):
                if not v[i] == None:
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

        #print "Suggested epsilon=%f, Using epsilon=%f, num_nodes=%f" % (suggested_epsilon, ep, num_nodes)
        return root_node, ep, l, num_nodes

    def BuildTree(self, node, H, isRoot=False):
        """
        Builds the preprocessing (semi) tree recursively
        """

        if not isRoot: node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return

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
            new_st = st.children[a]

            # Reward is just the mean added to a multiple of the variance at that point
            mean = self.gp.GPMean(x_next.history.locations, x_next.history.measurements, x_next.physical_state,
                                  weights=new_st.weights)
            var = new_st.variance
            r = self.reward_analytical(mean, math.sqrt(var))

            # Future reward
            f = self.EstimateV_tilde(T, l, gamma, x_next, new_st) + r

            if (f > vBest):
                aBest = a
                vBest = f

        return vBest, aBest

    def EstimateV_tilde(self, T, l, gamma, x, new_st):
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

        #num_partitions = new_st.n + 2  # number of partitions INCLUDING the tail ends

        # todo check
        n = 10

        sams = np.random.normal(mu, sd, n)

        rrr = [self.EstimateV(T - 1, l,gamma,self.TransitionH(x, sam),
                                   new_st)[0] + self.reward_sampled(sam) for sam in sams]
        avg = np.mean(rrr)

        return avg

        #return vAccum

    def MCTSRollout(self, action_node, st, T, l):

        if T == 0: return (0, 0, 0)
        assert not action_node.saturated, "Exploring saturated action node"

        # Select action that has the greatest upper bound (TODO: make sure there are still leaves in that branch)
        highest_upper = -float('inf')
        best_a = None
        for a, bounds in action_node.BoundsChildren.iteritems():
            if action_node.ChanceChildren[a].saturated: continue
            if highest_upper < bounds[1]: best_a = a
            highest_upper = max(highest_upper, bounds[1])

        new_semi_tree = st.children[best_a]

        # Select observation that has the greatest WEIGHTED error
        obs_node = action_node.ChanceChildren[best_a]
        highest_variance = -0.5
        most_uncertain_node_index = None
        for i in xrange(obs_node.num_partitions):
            if not (obs_node.ActionChildren[i] == None) and obs_node.ActionChildren[i].saturated: continue
            if (obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0]) * obs_node.IntervalWeights[
                i] > highest_variance:
                most_uncertain_node_index = i
                highest_variance = (obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0]) * \
                                   obs_node.IntervalWeights[i]

        i = most_uncertain_node_index
        # If observation is leaf, then we expand:
        if obs_node.ActionChildren[i] == None:

            new_action_node = MCTSActionNode(TransitionH(obs_node.augmented_state, obs_node.ObservationValue[i]),
                                             new_semi_tree, self, l)
            obs_node.ActionChildren[i] = new_action_node

            num_nodes_expanded = new_action_node.SkeletalExpand()
            # Update upper and lower bounds on this observation node
            lower, upper = new_action_node.Eval()
            obs_node.BoundsChildren[i] = (
            max(obs_node.BoundsChildren[i][0], lower), min(obs_node.BoundsChildren[i][1], upper))

        else:  # Observation has already been made, expand further

            lower, upper, num_nodes_expanded = self.MCTSRollout(obs_node.ActionChildren[i], new_semi_tree, T - 1, l)
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


class MCTSActionNode:
    """
    """

    mini_epsilon = 10 ** -8

    def __init__(self, augmented_state, semi_tree, treeplan, l):
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan
        self.lamb = l

        self.Saturated = False
        self.ChanceChildren = dict()
        self.BoundsChildren = dict()

    def Eval(self):
        """
        Evaluate upper and lower bounds of this action node
        """

        if len(self.BoundsChildren) == 0: return (-MCTSActionNode.mini_epsilon, MCTSActionNode.mini_epsilon)

        max_upper = -float('inf')
        max_lower = -float('inf')
        for a, b in self.BoundsChildren.iteritems():
            max_upper = max(b[1], max_upper)
            max_lower = max(b[0], max_lower)

        self.max_upper = max_upper
        self.max_lower = max_lower

        return max_lower, max_upper

    def SkeletalExpand(self):
        """ Builds observation nodes for every action
        """

        num_nodes_expanded = 1
        for a, semi_child in self.semi_tree.children.iteritems():
            c = MCTSChanceNode(TransitionP(self.augmented_state, a), semi_child, self.treeplan, self.lamb)
            num_nodes_expanded += c.SkeletalExpand()
            self.ChanceChildren[a] = c
            self.BoundsChildren[a] = c.Eval()

        # self.DetermineDominance()
        self.DetermineSaturation()
        return num_nodes_expanded

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

        self.saturated = allSat


class MCTSChanceNode:
    def __init__(self, augmented_state, semi_tree, treeplan, l):
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan
        self.lamb = l

        # Number of partitions INCLUDING tails
        if self.semi_tree.n == 0:
            self.num_partitions = 1  # MLE case
        else:
            self.num_partitions = self.semi_tree.n + 2

        self.saturated = False
        self.numchild_unsaturated = self.num_partitions
        self.mu = self.treeplan.gp.GPMean(augmented_state.history.locations, augmented_state.history.measurements,
                                          augmented_state.physical_state, weights=semi_tree.weights)

        # Pointer too children action selection nodes. "None" = this observation has not been expanded.
        self.ActionChildren = [None] * self.num_partitions
        # Array of (lower, upper) tuple. Includes bounds which are due to Lipschitz constraints.
        self.BoundsChildren = [(-float('inf'), float('inf'))] * self.num_partitions
        # Range of observations for this partition
        self.ObservationBounds = [None] * self.num_partitions
        # Value of observation that we take from this partition
        self.ObservationValue = [None] * self.num_partitions
        # Weight of each interval
        self.IntervalWeights = [None] * self.num_partitions

        #################################################
        # Compute partition information when NOT mle
        #################################################
        if self.num_partitions > 1:

            # Initialize variables
            kxi = semi_tree.lipchitz
            mu = self.mu
            sd = math.sqrt(semi_tree.variance)
            k = semi_tree.k

            if semi_tree.n > 0: width = 2.0 * k * sd / semi_tree.n
            for i in xrange(2, self.num_partitions):
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

    def Eval(self):
        """
        Evaluate upper and lower bounds of this chance node (weighted)
        """

        lower = 0.0
        upper = 0.0

        for i in xrange(len(self.BoundsChildren)):
            lower += (self.BoundsChildren[i][0] + self.treeplan.reward_sampled(self.ObservationValue[i])) * \
                     self.IntervalWeights[i]
            upper += (self.BoundsChildren[i][1] + self.treeplan.reward_sampled(self.ObservationValue[i])) * \
                     self.IntervalWeights[i]

        # Update reward
        # lower += self.mu - self.semi_tree.true_error
        # upper += self.mu + self.semi_tree.true_error
        r = self.treeplan.reward_analytical(self.mu, math.sqrt(self.semi_tree.variance))
        lower += r - self.semi_tree.true_error
        upper += r + self.semi_tree.true_error

        assert (lower <= upper), "Lower > Upper!, %s, %s" % (lower, upper)

        return lower, upper

    def UpdateChildrenBounds(self, index_updated):
        """ Update bounds of OTHER children while taking into account lipschitz constraints
        @param index_updated: index of child whose bound was just updated
        """

        lip = self.semi_tree.lipchitz

        assert self.BoundsChildren[index_updated][0] <= self.BoundsChildren[index_updated][1], "%s, %s" % (
        self.BoundsChildren[index_updated][0], self.BoundsChildren[index_updated][1])
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
            self.BoundsChildren[i][0] <= self.BoundsChildren[i][1]), "lower bound greater than upper bound %f, %f" % (
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
            self.BoundsChildren[i][0] <= self.BoundsChildren[i][1]), "lower bound greater than upper bound %f, %f" % (
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

        target = int(math.floor(self.num_partitions / 2))
        num_nodes_expanded += self.SkeletalExpandHere(target)
        self.UpdateChildrenBounds(target)
        return num_nodes_expanded

    def SkeletalExpandHere(self, index_to_expand):
        """ Expand given node at a particular index
        """
        num_nodes_expanded = 0
        assert self.ActionChildren[index_to_expand] == None, "Node already expanded"
        self.ActionChildren[index_to_expand] = MCTSActionNode(
            TransitionH(self.augmented_state, self.ObservationValue[index_to_expand]), self.semi_tree, self.treeplan,
            self.lamb)
        num_nodes_expanded += self.ActionChildren[index_to_expand].SkeletalExpand()
        lower, upper = self.ActionChildren[index_to_expand].Eval()
        assert lower <= upper
        # print lower, upper, self.BoundsChildren[index_to_expand]
        self.BoundsChildren[index_to_expand] = (
        max(self.BoundsChildren[index_to_expand][0], lower), min(self.BoundsChildren[index_to_expand][1], upper))
        # print self.BoundsChildren[index_to_expand]
        assert self.BoundsChildren[index_to_expand][0] <= self.BoundsChildren[index_to_expand][1]
        self.UpdateChildrenBounds(index_to_expand)

        if self.ActionChildren[index_to_expand].saturated:
            self.numchild_unsaturated -= 1
            if self.numchild_unsaturated == 0: self.saturated = True

        return num_nodes_expanded


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
