import copy
import gc
import math
import random

import numpy as np
from scipy.stats import norm

from MacroActionGenerator import GenerateSimpleMacroactions
from qEI import qEI
from SampleFunctionBuilder import GetNumberOfSamples
from src.r_qei import newQEI


class TreePlan:
    def __init__(self, batch_size, grid_domain, horizon, grid_gap, num_samples, gaussian_process, model,
                 macroaction_set=None,
                 max_nodes=None,
                 beta=0.0):

        self.model = model

        self.batch_size = batch_size

        self.H = horizon
        # Number of observations/samples generated for every node
        self.samples_per_stage = num_samples

        # Problem parameters
        self.grid_gap = grid_gap
        self.macroaction_set = macroaction_set

        # only for simulated
        """
        if macroaction_set is None:
            self.macroaction_set = GenerateSimpleMacroactions(self.batch_size, self.grid_gap)
        """
        self.grid_domain = grid_domain
        self.gp = gaussian_process
        self.max_nodes = float("inf") if max_nodes is None else max_nodes
        self.beta = beta

        self.l1 = 0
        # unused
        self.l2 = lambda sigma: 1

        self.reward_analytical = lambda mu, sigma: (
            self.AcquizitionFunction(mu, sigma) - self.batch_size * self.gp.mean_function)

        # unused
        self.reward_sampled = lambda f: 0

    # heuristic
    # we use batch UCB version from Erik
    # todo check that we do not add noise twice
    def AcquizitionFunction(self, mu, sigma):
        if self.beta == 0:
            return np.sum(mu)
        exploration_matrix = np.identity(sigma.shape[0]) + (1 / self.gp.covariance_function.noise_variance) * sigma
        # print np.sum(mu), math.log(np.linalg.det(exploration_matrix))
        return np.sum(mu) + self.beta * math.log(np.linalg.det(exploration_matrix))

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
        return self.ComputeVMLE(T - 1, self.TransitionH(x, mu), new_st)[0]

    def new_qEI(self, x_0):
        # x_0 stores a 2D np array of k points with history

        best_action = None
        best_expected_improv = -float('inf')

        qei = newQEI(length_scale=self.gp.length_scale, signal_variance=self.gp.signal_variance,
                     noise_variance=self.gp.noise,
                     locations=x_0.history.locations, Y=x_0.history.measurements - np.mean(x_0.history.measurements))

        next_states = self.GetNextAugmentedStates(x_0)
        if not next_states:
            raise Exception("qEI could not move from  location " + str(x_0.physical_state))

        for x_next in next_states:
            # x_next = self.TransitionP(x_0, a)

            expectedImprov = qei.acquisition(x_next.physical_state)

            # comparison
            if expectedImprov >= best_expected_improv:
                best_expected_improv = expectedImprov
                best_action = x_next

        return best_expected_improv, best_action, len(next_states)

    def qEI(self, x_0, eps=10 ** (-5)):
        # x_0 stores a 2D np array of k points with history
        max_measurement = max(x_0.history.measurements)

        best_action = None
        best_expected_improv = - float("inf")
        best_expected_improv = -1.0

        # valid_actions = self.GetValidActionSet(x_0.physical_state)
        # next_states = [self.TransitionP(x_0, a) for a in valid_actions]

        next_states = self.GetNextAugmentedStates(x_0)
        if not next_states:
            raise Exception("qEI could not move from   location " + str(x_0.physical_state))

        chol = self.gp.Cholesky(x_0.history.locations)

        for x_next in next_states:
            # x_next = self.TransitionP(x_0, a)

            Sigma = self.gp.GPVariance(locations=x_next.history.locations, current_location=x_next.physical_state,
                                       cholesky=chol)
            weights = self.gp.GPWeights(locations=x_next.history.locations, current_location=x_next.physical_state,
                                        cholesky=chol)
            mu = self.gp.GPMean(measurements=x_next.history.measurements, weights=weights)

            expectedImprov = qEI(Sigma, eps, mu, max_measurement, self.batch_size)

            # comparison
            if expectedImprov >= best_expected_improv:
                best_expected_improv = expectedImprov
                best_action = x_next

        return best_expected_improv, best_action, len(next_states)

    # given a list of macroactions, find the set of unique points at step i
    def GetSetOfNextPoints(self, available_states, step):
        # available states should be a list of type AugmentedState
        next_points_tuples = map(tuple, [next_state.physical_state[step, :] for next_state in available_states])
        # a set of tuples
        next_points = set(next_points_tuples)
        return map(lambda x: np.atleast_2d(x), list(next_points))

    def PI(self, x_0):

        best_observation = max(x_0.history.measurements)

        next_states = self.GetNextAugmentedStates(x_0)

        vBest = 0.0
        xBest = next_states[0]

        current_locations = x_0.history.locations
        current_chol = self.gp.Cholesky(x_0.history.locations)

        for x_next in next_states:

            next_physical_state = x_next.physical_state
            var = self.gp.GPVariance(locations=current_locations, current_location=next_physical_state,
                                     cholesky=current_chol)
            sigma = math.sqrt(var[0, 0])

            weights = self.gp.GPWeights(locations=current_locations, current_location=next_physical_state,
                                        cholesky=current_chol)
            mu = self.gp.GPMean(measurements=x_0.history.measurements, weights=weights)[0]

            Z = (mu - best_observation) / sigma

            probImprovement = norm.cdf(x=Z, loc=0, scale=1.0)
            # probImprovement = 1.0 - norm.cdf(x=best_observation, loc=mu, scale=sigma)

            if probImprovement >= vBest:
                vBest = probImprovement
                xBest = x_next

        return vBest, xBest, -1

    def EI(self, x_0):

        best_observation = max(x_0.history.measurements)

        next_states = self.GetNextAugmentedStates(x_0)

        vBest = 0.0
        xBest = next_states[0]

        current_locations = x_0.history.locations
        current_chol = self.gp.Cholesky(x_0.history.locations)

        for x_next in next_states:

            next_physical_state = x_next.physical_state
            var = self.gp.GPVariance(locations=current_locations, current_location=next_physical_state,
                                     cholesky=current_chol)
            sigma = math.sqrt(var[0, 0])

            weights = self.gp.GPWeights(locations=current_locations, current_location=next_physical_state,
                                        cholesky=current_chol)
            mu = self.gp.GPMean(measurements=x_0.history.measurements, weights=weights)[0]

            Z = (mu - best_observation) / sigma
            expectedImprov = (mu - best_observation) * norm.cdf(x=Z, loc=0, scale=1.0) + sigma * norm.pdf(x=Z, loc=0,
                                                                                                          scale=1.0)
            if expectedImprov >= vBest:
                vBest = expectedImprov
                xBest = x_next

        return vBest, xBest, -1

    def BUCB(self, x_0, t):

        tolerance_eps = 10 ** (-8)

        available_states = self.GetNextAugmentedStates(x_0)

        if not available_states:
            raise Exception("BUCB could not move from  location " + str(x_0.physical_state))

        # first_points = self.GetSetOfNextPoints(available_states, 0)

        domain_size = 145
        delta = 0.1
        beta_multiplier = 0.2

        history_locations = x_0.history.locations

        # get the mu values for the first point
        mu_values = {}
        current_chol = self.gp.Cholesky(x_0.history.locations)
        first_points = self.GetSetOfNextPoints(available_states, 0)
        for first_point in first_points:
            weights = self.gp.GPWeights(locations=history_locations, current_location=first_point,
                                        cholesky=current_chol)
            mu = self.gp.GPMean(measurements=x_0.history.measurements, weights=weights)
            mu_values[tuple(first_point[0])] = mu

        for num_steps in range(self.batch_size):
            value_dict= {}
            best_next_value = - float("inf")
            for next_state in available_states:
                current_locations = np.append(history_locations, next_state.physical_state[:num_steps, :], axis=0)

                current_chol = self.gp.Cholesky(current_locations)
                next_point = next_state.physical_state[num_steps: num_steps+1, :]
                Sigma = self.gp.GPVariance(locations=current_locations, current_location=next_point,
                                           cholesky=current_chol)

                first_point = next_state.physical_state[0:1, :]
                mu = mu_values[tuple(first_point[0])]
                iteration = self.batch_size * t + num_steps + 1
                beta_t1 = 2 * beta_multiplier * math.log(domain_size * (iteration**2) * (math.pi ** 2) / (6 * delta))
                predicted_val = mu[0] + math.sqrt(beta_t1) * math.sqrt(Sigma[0, 0])

                best_next_value = max(predicted_val, best_next_value)

                value_dict[tuple(map(tuple, next_state.physical_state))] = predicted_val

            available_states = [next_state for next_state in available_states if
                                abs(value_dict[tuple(map(tuple, next_state.physical_state))]
                                    - best_next_value) < tolerance_eps]

        return -1.0, random.choice(available_states), -1.0

    def BUCB_PE(self, x_0, t):

        tolerance_eps = 10 ** (-8)

        available_states = self.GetNextAugmentedStates(x_0)

        if not available_states:
            raise Exception("BUCB-PE could not move from  location " + str(x_0.physical_state))

        domain_size = 145
        delta = 0.1
        t_squared = (t + 1) ** 2
        beta_t1 = 2 * math.log(domain_size * t_squared * (math.pi ** 2) / (6 * delta))

        best_current_measurement = - float("inf")
        history_locations = x_0.history.locations
        current_chol = self.gp.Cholesky(x_0.history.locations)

        predict_val_dict = {}
        # first step is ucb
        for next_state in available_states:
            first_point = next_state.physical_state[0:1, :]
            Sigma = self.gp.GPVariance(locations=history_locations, current_location=first_point,
                                       cholesky=current_chol)
            weights = self.gp.GPWeights(locations=history_locations, current_location=first_point,
                                        cholesky=current_chol)
            mu = self.gp.GPMean(measurements=x_0.history.measurements, weights=weights)

            predicted_val = mu[0] + math.sqrt(beta_t1) * math.sqrt(Sigma[0, 0])

            predict_val_dict[tuple(first_point[0])] = predicted_val
            if predicted_val > best_current_measurement:
                best_current_measurement = predicted_val

        # the states with selected several points according to batch construction
        available_states = [next_state for next_state in available_states if
                            abs(predict_val_dict[tuple(next_state.physical_state[0, :])]
                                - best_current_measurement) < tolerance_eps]

        # Pure exploration part
        for num_steps in range(1, self.batch_size):
            sigma_dict = {}
            best_next_sigma = - float("inf")

            for next_state in available_states:
                current_locations = np.append(history_locations, next_state.physical_state[:num_steps, :], axis=0)

                current_chol = self.gp.Cholesky(current_locations)
                next_point = next_state.physical_state[num_steps: num_steps + 1, :]
                # current_points = self.GetSetOfNextPoints(available_states, num_steps)
                sigma = self.gp.GPVariance(locations=current_locations, current_location=next_point,
                                           cholesky=current_chol)[0, 0]
                best_next_sigma = max(sigma, best_next_sigma)
                sigma_dict[tuple(map(tuple, next_state.physical_state))] = sigma

            available_states = [next_state for next_state in available_states if
                                abs(sigma_dict[tuple(map(tuple, next_state.physical_state))]
                                    - best_next_sigma) < tolerance_eps]

        return -1.0, random.choice(available_states), -1.0

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
        root_action_node = MCTSActionNode(x_0, root_node, self, lamb)
        print "MCTS max nodes:", max_nodes, "Skeletal Expansion"
        # Expand tree
        total_nodes_expanded = root_action_node.SkeletalExpand()
        gc.collect()
        print "Performing search..."

        number_of_iterations = 800 if H == 4 else 1500
        # number_of_iterations = 1 if H == 4 else 1
        counter = 0
        # TODO: Set a proper termination condition
        # whilre resources permit
        while not root_action_node.saturated and total_nodes_expanded < max_nodes:
            print counter, H
            lower, upper, num_nodes_expanded = self.ConstructTree(root_action_node, root_node, H, lamb)
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

    # used only for simulated
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
        return True

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


# TODO note history includes current state
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
            fake_action = np.zeros(next_physical_state.shape)
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

    def SkeletalExpand(self):
        """ Expand only using observations at the edges
        """

        num_nodes_expanded = 0

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


# TODO note history locations includes current state
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
