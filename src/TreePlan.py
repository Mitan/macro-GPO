import gc
import math
import numpy as np

from src.AnytimeNode import MCTSActionNode
from src.core.SemiTree import SemiTree
from src.core.SemiState import SemiState
from src.core.Transitions import TransitionH, TransitionP
from src.Utils import ToTuple


class TreePlan:
    def __init__(self, batch_size, domain_descriptor, horizon, num_samples, gaussian_process, model,
                 macroaction_set=None,
                 max_nodes=None,
                 beta=0.0):

        self.model = model
        self.domain_descriptor = domain_descriptor
        self.batch_size = batch_size

        self.H = horizon
        # Number of observations/samples generated for every node
        self.samples_per_stage = num_samples

        # Problem parameters
        self.macroaction_set = macroaction_set

        self.gp = gaussian_process
        self.max_nodes = float("inf") if max_nodes is None else max_nodes
        self.beta = beta

        self.reward_analytical = lambda mu, sigma: (
                self.AcquizitionFunction(mu, sigma) - self.batch_size * self.gp.mean_function)

    def AcquizitionFunction(self, mu, sigma):
        if self.beta == 0:
            return np.sum(mu)
        exploration_matrix = np.identity(sigma.shape[0]) + (1 / self.gp.noise_variance) * sigma
        return np.sum(mu) + self.beta * math.log(np.linalg.det(exploration_matrix))

    def GetNextAugmentedStates(self, current_augmented_state):

        current_location = current_augmented_state.physical_state[-1, :]

        new_physical_states = self.model.GetSelectedMacroActions(current_location)

        if not new_physical_states:
            return []

        next_states = []

        for next_p_state in new_physical_states:
            next_st = TransitionP(current_augmented_state, next_p_state)
            next_states.append(next_st)

        return next_states

    def StochasticFull(self, x_0, H):

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        root_ss = SemiState(physical_state, past_locations)
        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        future_dict = {}

        Vapprox, Xapprox = self.ComputeVRandom(H, x_0, root_node, future_dict)
        if math.isinf(Vapprox):
            raise Exception("Exact full H = " + str(H) + " could not move from  location " + str(x_0.physical_state))

        future_steps = []
        current_point  = ToTuple(Xapprox.physical_state)

        while current_point is not None:
            future_steps.append(current_point)
            current_point = future_dict[ToTuple(current_point)]
        print H, len(future_steps)
        return Vapprox, Xapprox, future_steps, -1

    def ComputeVRandom(self, T, x, st, future_steps):

        if T == 0:
            return 0, np.zeros((self.batch_size, 2))

        next_states = self.GetNextAugmentedStates(x)
        if not next_states:
            return -float("inf"), np.zeros((self.batch_size, 2))

        vBest = -float("inf")
        xBest = next_states[0]
        for x_next in next_states:
            next_physical_state = x_next.physical_state
            new_st = st.children[ToTuple(next_physical_state)]

            # Reward is just the mean added to a multiple of the variance at that point

            mean = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)
            var = new_st.variance

            r = self.reward_analytical(mean, var)

            # Future reward
            f = self.ComputeQRandom(T, x_next, new_st, future_steps) + r

            if f > vBest:
                xBest = x_next
                vBest = f

        future_steps[ToTuple(x.physical_state)] = xBest.physical_state

        return vBest, xBest

    def ComputeQRandom(self, T, x, new_st, future_steps):

        if T == 1:

            future_steps[ToTuple(x.physical_state)] = None
            return 0

        mu = self.gp.GPMean(measurements=x.history.measurements, weights=new_st.weights)

        sd = new_st.variance

        number_of_samples = self.samples_per_stage
        sams = np.random.multivariate_normal(mu, sd, number_of_samples)

        rrr = [self.ComputeVRandom(T - 1,
                                   TransitionH(x, sam),
                                   new_st,
                                   future_steps)[0] for sam in sams]
        avg = np.mean(rrr)

        return avg

    def AnytimeAlgorithm(self, epsilon, x_0, H, anytime_num_iterations, max_nodes=10 ** 15):
        print "ANYTIME " + str(H)
        print "Preprocessing weight spaces..."

        # by default physical state length is self.batch_size
        # but for the first step it is equal to 1, since it is just agent's position

        # x_0.history.locations includes current location

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        root_ss = SemiState(physical_state, past_locations)

        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)
        self.PreprocessLipchitz(root_node)

        lamb = epsilon
        print "lambda is " + str(lamb)

        # node d_0, where we have actions
        root_action_node = MCTSActionNode(augmented_state=x_0, semi_tree=root_node, treeplan=self, l=lamb, level=H)
        print "MCTS max nodes:", max_nodes, "Skeletal Expansion"

        # Expand tree
        total_nodes_expanded = root_action_node.SkeletalExpand()
        gc.collect()
        print "Performing search..."

        counter = 0

        while not root_action_node.saturated and total_nodes_expanded < max_nodes:
            print counter, H
            _, _, num_nodes_expanded = self.ConstructTree(root_action_node, root_node, H, lamb)
            total_nodes_expanded += num_nodes_expanded
            counter += 1
            gc.collect()
            gc.collect()
            gc.collect()
            if counter > anytime_num_iterations:
                break
        print "counter is " + str(counter)

        best_lower = -float('inf')
        for a, cc in root_action_node.BoundsChildren.iteritems():
            print a, cc
            if best_lower < cc[0]:
                best_a = a
                best_lower = cc[0]

        if math.isinf(best_lower):
            raise Exception("Anytime for " + str(H) + " could not move from  location " + str(x_0.physical_state))

        print best_lower, np.asarray(best_a)

        print "Total nodes expanded %d" % total_nodes_expanded
        return root_action_node.BoundsChildren[best_a], np.asarray(best_a), total_nodes_expanded

    def BuildTree(self, node, H, isRoot=False):
        """
        Builds the preprocessing (semi) tree recursively
        """
        if not isRoot:
            node.ComputeWeightsAndVariance(self.gp)

        if H == 0:
            return

        cur_physical_state = node.ss.physical_state
        new_locations = np.append(node.ss.locations, np.atleast_2d(cur_physical_state), 0)
        cholesky = self.gp.Cholesky(new_locations)

        current_location = cur_physical_state[-1, :]

        new_physical_states = self.model.GetSelectedMacroActions(current_location)

        for new_physical_state in new_physical_states:
            # Get new semi state

            new_ss = SemiState(new_physical_state, new_locations)

            # Build child subtree
            new_st = SemiTree(new_ss, chol=cholesky)
            node.AddChild(new_physical_state, new_st)
            new_st.ComputeWeightsAndVariance(self.gp)
            self.BuildTree(new_st, H - 1)

    def PreprocessLipchitz(self, node):

        # Base case
        if len(node.children) == 0:
            lip = 0

        else:
            # Recursive case
            vmax = 0
            for _, c in node.children.iteritems():
                self.PreprocessLipchitz(c)
                alpha = np.linalg.norm(c.weights, ord='fro')
                av = c.lipchitz * math.sqrt(1 + alpha ** 2) + alpha * math.sqrt(self.batch_size)
                vmax = np.maximum(vmax, av)
                lip = vmax

        node.lipchitz = lip

    def ConstructTree(self, action_node, st, T, l):

        if T == 0:
            print "blea"
            return 0, 0, 1
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
        # we have approaching the leaf of the whole tree

        if T > 1:
            highest_variance = -0.5
            most_uncertain_node_index = 0
            for i in xrange(obs_node.num_samples):
                if not (obs_node.ActionChildren[i] is None) and obs_node.ActionChildren[i].saturated:
                    continue
                current_variance = obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0]
                if current_variance > highest_variance:
                    most_uncertain_node_index = i
                    highest_variance = current_variance

            i = most_uncertain_node_index
            # nothing to expand

            # If observation is leaf, then we expand:
            if obs_node.ActionChildren[i] is None:
                new_action_node = MCTSActionNode(
                    augmented_state=TransitionH(obs_node.augmented_state, obs_node.ObservationValue[i]),
                    semi_tree=new_semi_tree, treeplan=self, l=l, level=T - 1)
                obs_node.ActionChildren[i] = new_action_node

                num_nodes_expanded = new_action_node.SkeletalExpand()
                # Update upper and lower bounds on this observation node
                lower, upper = new_action_node.Eval()
                obs_node.BoundsChildren[i] = (
                    max(obs_node.BoundsChildren[i][0], lower), min(obs_node.BoundsChildren[i][1], upper))

            else:  # Observation has already been made, expand further
                lower, upper, num_nodes_expanded = self.ConstructTree(obs_node.ActionChildren[i], new_semi_tree, T - 1,
                                                                      l)
                obs_node.BoundsChildren[i] = (
                    max(lower, obs_node.BoundsChildren[i][0]), min(upper, obs_node.BoundsChildren[i][1]))
        else:
            lower, upper, num_nodes_expanded = (0, 0, 1)
            # has only one child
            i = 0

        obs_node.UpdateChildrenBounds(i)
        lower, upper = obs_node.Eval()
        assert (lower <= upper)
        action_node.BoundsChildren[best_a] = (
            max(action_node.BoundsChildren[best_a][0], lower), min(action_node.BoundsChildren[best_a][1], upper))

        if obs_node.ActionChildren[i].saturated:
            obs_node.numchild_unsaturated -= 1
            if obs_node.numchild_unsaturated == 0:
                obs_node.saturated = True

        action_node.DetermineSaturation()

        return action_node.Eval() + (num_nodes_expanded,)

