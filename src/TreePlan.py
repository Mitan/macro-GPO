import gc
import math
import numpy as np


from SampleFunctionBuilder import GetNumberOfSamples
from src.AnytimeNode import MCTSActionNode
from src.core.SemiTree import SemiTree
from src.core.SemiState import SemiState
from src.core.Transitions import TransitionH, TransitionP
from src.Utils import ToTuple, EI_Acquizition_Function

from src.methods.BBO_LP import method_LP
from src.methods.BUCB import method_BUCB
from src.methods.BUCB_PE import method_BUCB_PE
from src.methods.EI import method_EI
from src.methods.PI import method_PI
from src.methods.qEI import method_qEI
# from src.methods.qEI_R import method_qEI_R


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
        # self.grid_gap = grid_gap
        self.macroaction_set = macroaction_set

        # only for simulated
        """
        if macroaction_set is None:
            self.macroaction_set = GenerateSimpleMacroactions(self.batch_size, self.grid_gap)
        """
        # self.grid_domain = grid_domain
        self.gp = gaussian_process
        self.max_nodes = float("inf") if max_nodes is None else max_nodes
        self.beta = beta

        # self.l1 = 0
        # unused
        # self.l2 = lambda sigma: 1

        self.reward_analytical = lambda mu, sigma: (
                self.AcquizitionFunction(mu, sigma) - self.batch_size * self.gp.mean_function)

        # unused
        # self.reward_sampled = lambda f: 0

    # heuristic
    # we use batch UCB version from Erik
    # todo check that we do not add noise twice
    def AcquizitionFunction(self, mu, sigma):
        if self.beta == 0:
            return np.sum(mu)
        exploration_matrix = np.identity(sigma.shape[0]) + (1 / self.gp.covariance_function.noise_variance) * sigma
        # print np.sum(mu), math.log(np.linalg.det(exploration_matrix))
        return np.sum(mu) + self.beta * math.log(np.linalg.det(exploration_matrix))

    def RolloutFiniteBudget(self, x_0, H, gamma):

        if H == 1:
            return self.EI(x_0)

        physical_state = x_0.physical_state
        physical_state_size = physical_state.shape[0]
        past_locations = x_0.history.locations[: -physical_state_size, :]

        root_ss = SemiState(physical_state, past_locations)

        root_node = SemiTree(root_ss)
        self.BuildTree(root_node, H, isRoot=True)

        Vapprox, Xapprox = self.ComputeURollout(T=H, x=x_0, st=root_node, gamma=gamma)
        if math.isinf(Vapprox):
            raise Exception("MLE could not move from  location " + str(x_0.physical_state))

        return Vapprox, Xapprox, -1

    # calculate improvement
    def RolloutAcquizition(self, current_value, augmented_state):
        max_found_value = max(augmented_state.history.measurements)
        return max(0, current_value - max_found_value)

    def ComputeURollout(self, T, x, st, gamma):
        next_states = self.GetNextAugmentedStates(x)
        if not next_states:
            return -float("inf"), np.zeros((self.batch_size, 2))

        vBest = - float("inf")
        xBest = next_states[0]
        for x_next in next_states:

            # x_next = self.TransitionP(x, a)
            next_physical_state = x_next.physical_state

            # cannot tuple augmented state, need to use physical state here
            new_st = st.children[ToTuple(next_physical_state)]

            mean = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)

            r = self.RolloutAcquizition(current_value=mean, augmented_state=x)

            # Future reward
            f = r + gamma * self.ComputeHRollout(T=T - 1, x=TransitionH(x_next, mean), st=new_st, gamma=gamma)

            if f > vBest:
                xBest = x_next
                vBest = f
        # xBest is augmented state
        return vBest, xBest

    def ComputeHRollout(self, T, x, st, gamma):
        # action selected by PI policy
        # _, x_next, _ = self.PI(x)
        _, x_next, _ = self.EI(x)
        next_physical_state = x_next.physical_state
        new_st = st.children[ToTuple(next_physical_state)]
        mu = self.gp.GPMean(measurements=x_next.history.measurements, weights=new_st.weights)
        sigma = new_st.variance
        if T == 1:
            best_observation = max(x.history.measurements)
            return EI_Acquizition_Function(mu=mu, sigma=sigma, best_observation=best_observation)

        r = self.RolloutAcquizition(current_value=mu, augmented_state=x)

        return r + gamma * self.ComputeHRollout(T=T - 1, x=TransitionH(x_next, mu), st=new_st, gamma=gamma)

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
        # TODO for road create a fake action a = [0. 0. 0.] and then manually
        # TODO fix bad design
        # set physical state for every new augmented state
        # or create a new function TransitionP for setting augmented state

        # this is for real-world
        current_location = current_augmented_state.physical_state[-1, :]
        # print "H = " + str(T)
        # print current_location

        # new_physical_states = self.model.GenerateAllRoadMacroActions(current_location, self.batch_size)
        new_physical_states = self.model.GetSelectedMacroActions(current_location)

        # if we can't move further in this direction, then stop and return
        """
        if not new_physical_states:
            return -float("inf"), np.zeros((self.batch_size, 2))
        """
        if not new_physical_states:
            return []

        # print new_physical_states
        # fake_action = np.zeros(new_physical_states[0].shape)
        next_states = []

        for next_p_state in new_physical_states:
            next_st = TransitionP(current_augmented_state, next_p_state)
            # next_st.physical_state = next_p_state
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
        for x_next in next_states:

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
        # mu = self.gp.GPMean(locations=x.history.locations, measurements=x.history.measurements,
        # current_location=x.physical_state, cholesky=new_st.cholesky)
        return self.ComputeVMLE(T - 1, TransitionH(x, mu), new_st)[0]

    def new_qEI(self, x_0):
        return method_qEI_R(x_0=x_0, gp=self.gp, next_states=self.GetNextAugmentedStates(x_0))

    def qEI(self, x_0):
        return method_qEI(x_0=x_0, next_states=self.GetNextAugmentedStates(x_0),
                          gp=self.gp, batch_size=self.batch_size)

    def PI(self, x_0):
        return method_PI(x_0=x_0, gp=self.gp, next_states=self.GetNextAugmentedStates(x_0))

    def EI(self, x_0):
        return method_EI(x_0=x_0, gp=self.gp, next_states=self.GetNextAugmentedStates(x_0))

    def LP(self, x_0, t):
        return method_LP(x_0=x_0, available_states=self.GetNextAugmentedStates(x_0), gp=self.gp,
                         batch_size=self.batch_size)

    def BUCB(self, x_0, t):
        return method_BUCB(x_0=x_0, gp=self.gp, t=t,
                           available_states=self.GetNextAugmentedStates(x_0),
                           batch_size=self.batch_size,
                           domain_size=self.domain_descriptor.domain_size)

    def BUCB_PE(self, x_0, t):
        return method_BUCB_PE(x_0=x_0, gp=self.gp, t=t,
                              available_states=self.GetNextAugmentedStates(x_0),
                              batch_size=self.batch_size,
                              domain_size=self.domain_descriptor.domain_size)

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
        number_of_samples = self.samples_per_stage
        # sams = np.random.multivariate_normal(mu, sd, self.samples_per_stage)
        sams = np.random.multivariate_normal(mu, sd, number_of_samples)

        rrr = [self.ComputeVRandom(T - 1, TransitionH(x, sam),
                                   new_st)[0] for sam in sams]
        avg = np.mean(rrr)

        return avg

    def AnytimeAlgorithm(self, epsilon, x_0, H, max_nodes=10 ** 15):
        print "ANYTIME " + str(H)
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
        lamb = 5.0
        print "lambda is " + str(lamb)
        # node d_0, where we have actions
        root_action_node = MCTSActionNode(augmented_state=x_0, semi_tree=root_node, treeplan=self, l=lamb, level=H)
        print "MCTS max nodes:", max_nodes, "Skeletal Expansion"
        # Expand tree
        total_nodes_expanded = root_action_node.SkeletalExpand()
        gc.collect()
        print "Performing search..."

        number_of_iterations = 800 if H == 4 else 1500
        number_of_iterations = 600 if H == 4 else 1500
        number_of_iterations = 2000
        # number_of_iterations = 1500
        # number_of_iterations = 1 if H == 4 else 1
        counter = 0
        # TODO: Set a proper termination condition
        # whilre resources permit
        while not root_action_node.saturated and total_nodes_expanded < max_nodes:
            print counter, H
            _, _, num_nodes_expanded = self.ConstructTree(root_action_node, root_node, H, lamb)
            total_nodes_expanded += num_nodes_expanded
            counter += 1
            gc.collect()
            gc.collect()
            gc.collect()
            if counter > number_of_iterations:
                break
        print "counter is " + str(counter)
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
    """
    def Preprocess(self, physical_state, locations, H, suggested_epsilon):

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
    """
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

        # new_physical_states = self.model.GenerateAllRoadMacroActions(current_location, self.batch_size)
        new_physical_states = self.model.GetSelectedMacroActions(current_location)

        for new_physical_state in new_physical_states:
            # Get new semi state

            # new_physical_state = self.PhysicalTransition(cur_physical_state, a)
            new_ss = SemiState(new_physical_state, new_locations)

            # Build child subtree
            new_st = SemiTree(new_ss, chol=cholesky)
            node.AddChild(new_physical_state, new_st)
            new_st.ComputeWeightsAndVariance(self.gp)
            self.BuildTree(new_st, H - 1)

    def PreprocessLipchitz(self, node):

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

        # action_node.DetermineDominance()
        action_node.DetermineSaturation()

        return action_node.Eval() + (num_nodes_expanded,)

    # Hacks to overcome bad design
    """
    def TransitionP(self, augmented_state, action):
         return TransitionP(augmented_state, action)
    
    def TransitionH(self, augmented_state, measurement):
        return TransitionH(augmented_state, measurement)
    
    def PhysicalTransition(self, physical_state, action):
        return PhysicalTransition(physical_state, action)
    """

    
if __name__ == "__main__":
    pass
