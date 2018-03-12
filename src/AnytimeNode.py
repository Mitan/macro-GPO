import numpy as np

from src.core.Transitions import TransitionP, TransitionH


class MCTSActionNode:
    """
    """
    # d_t
    mini_epsilon = 10 ** -8

    def __init__(self, augmented_state, semi_tree, treeplan, l, level):
        self.level = level
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan
        # number of samples per stage at every child ObservationNode
        self.number_of_samples = self.treeplan.samples_per_stage

        self.lamb = l
        if level > 0:
            # is full?
            self.saturated = False
            # d_t + s_{t]
            self.ChanceChildren = dict()
            # Q_lower and Q_upper for each child
            self.BoundsChildren = dict()
            self.max_upper = -float('inf')
            self.max_lower = -float('inf')
        else:
            # is full?
            self.saturated = True
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
            c = MCTSObservationNode(augmented_state=next_augmented_state, semi_tree=semi_child, treeplan=self.treeplan,
                                    l=self.lamb,
                                    number_of_samples=self.number_of_samples, level=self.level)
            current_nodes = c.SkeletalExpand()
            num_nodes_expanded += current_nodes
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
    def __init__(self, augmented_state, semi_tree, treeplan, l, number_of_samples, level):
        self.augmented_state = augmented_state
        self.semi_tree = semi_tree
        self.treeplan = treeplan
        self.lamb = l
        self.level = level

        self.num_samples = number_of_samples

        self.saturated = False

        self.mu = self.treeplan.gp.GPMean(measurements=augmented_state.history.measurements, weights=semi_tree.weights)

        if self.level != 1:
            self.numchild_unsaturated = self.num_samples
            # Pointer too children action selection nodes. "None" = this observation has not been expanded.
            self.ActionChildren = [None] * self.num_samples
            # Array of (lower, upper) tuple. Includes bounds which are due to Lipschitz constraints.
            self.BoundsChildren = [(-float('inf'), float('inf'))] * self.num_samples

            mu = self.mu
            # sd = math.sqrt(semi_tree.variance)
            samples = np.random.multivariate_normal(mu, semi_tree.variance, self.num_samples)
            # samples = np.random.normal(mu, sd, self.num_samples)

            # cannot sort
            # self.ObservationValue = np.sort(samples, axis=None)
            self.ObservationValue = samples
        else:
            self.numchild_unsaturated = 1
            self.num_samples = 1
            self.ActionChildren = [None]
            # Array of (lower, upper) tuple. Includes bounds which are due to Lipschitz constraints.
            self.BoundsChildren = [0, 0]
            self.ObservationValue = np.array([self.mu])

    def Eval(self):
        """
        Evaluate upper and lower bounds of this chance node (weighted)
       they are Q_lower and Q_upper
        """
        r = self.treeplan.reward_analytical(self.mu, self.semi_tree.variance)
        # lower = 0.0
        # upper = 0.0
        # the same as num_samples
        number_of_children = len(self.BoundsChildren)
        if self.level > 1:
            lower = sum([childBound[0] for childBound in self.BoundsChildren]) / number_of_children
            upper = sum([childBound[1] for childBound in self.BoundsChildren]) / number_of_children
            lower += r - self.lamb
            upper += r + self.lamb
        else:
            lower = r
            upper = r
        assert (lower <= upper), "Lower > Upper!, %s, %s" % (lower, upper)

        return lower, upper

    def UpdateChildrenBounds(self, index_updated):
        """ Update bounds of OTHER children while taking into account lipschitz constraints
        @param index_updated: index of child whose bound was just updated
        """
        # can't update bounds if level = 1
        if self.level == 1:
            return
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
            augmented_state=TransitionH(self.augmented_state, self.ObservationValue[index_to_expand]),
            semi_tree=self.semi_tree, treeplan=self.treeplan,
            l=self.lamb, level=self.level - 1)

        # we can't expand further if level = 1
        if self.level > 1:
            num_nodes_expanded += self.ActionChildren[index_to_expand].SkeletalExpand()
            lower, upper = self.ActionChildren[index_to_expand].Eval()
            assert lower <= upper
            # print lower, upper, self.BoundsChildren[index_to_expand]
            self.BoundsChildren[index_to_expand] = (
                # tighten the bounds
                max(self.BoundsChildren[index_to_expand][0], lower),
                min(self.BoundsChildren[index_to_expand][1], upper))
            # print self.BoundsChildren[index_to_expand]
            assert self.BoundsChildren[index_to_expand][0] <= self.BoundsChildren[index_to_expand][1]
            self.UpdateChildrenBounds(index_to_expand)

        else:
            # we can't expand, but need to count this node
            num_nodes_expanded = 1
        if self.ActionChildren[index_to_expand].saturated:
            self.numchild_unsaturated -= 1
            if self.numchild_unsaturated == 0: self.saturated = True

        return num_nodes_expanded