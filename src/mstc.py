class MCTSActionNode:
	"""
	"""

	mini_epsilon = 10**-8

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
				cc.saturated = True # saturate all nodes which are dominated

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
			self.num_partitions = 1 # MLE case
		else:
			self.num_partitions = self.semi_tree.n + 2

		self.saturated = False
		self.numchild_unsaturated = self.num_partitions
		self.mu = self.treeplan.gp.GPMean(augmented_state.history.locations, augmented_state.history.measurements, augmented_state.physical_state, weights = semi_tree.weights)

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
				zLower = mu - sd * k + (i-2) * width
				zUpper = mu - sd * k + (i-1) * width
				self.ObservationBounds[i-1] = (zLower, zUpper)

				# Compute evaluation points
				self.ObservationValue[i-1] = 0.5 * (zLower + zUpper)

				# Compute weights
				self.IntervalWeights[i-1] = norm.cdf(x = zUpper, loc = mu, scale = sd) - norm.cdf(x = zLower, loc = mu, scale = sd)

			# Values for extremes
			rightLimit =  mu + k * sd
			leftLimit = mu - k * sd
			self.ObservationBounds[0] = (-float('inf'), leftLimit)
			self.ObservationBounds[-1] = (rightLimit, float('inf'))
			self.ObservationValue[0] = leftLimit
			self.ObservationValue[-1] = rightLimit
			self.IntervalWeights[0] =  norm.cdf(x = leftLimit, loc = mu, scale = sd)
			self.IntervalWeights[-1] =  1-norm.cdf(x = rightLimit, loc = mu, scale = sd)

			assert abs(sum(self.IntervalWeights) - 1) < 0.0001, "Area != 1, %f instead\n With number: %s " % (sum(self.IntervalWeights), str(self.IntervalWeights))

		else:
			#################################################
			# Compute partition information when using mle
			#################################################

			self.ObservationBounds[0] = (-float('inf'), float('inf'))
			self.ObservationValue[0] = self.mu
			self.IntervalWeights[0] =  1.0

	def Eval(self):
		"""
		Evaluate upper and lower bounds of this chance node (weighted)
		"""

		lower = 0.0
		upper = 0.0

		for i in xrange(len(self.BoundsChildren)):
			lower += (self.BoundsChildren[i][0] + self.treeplan.reward_sampled(self.ObservationValue[i])) * self.IntervalWeights[i]
			upper += (self.BoundsChildren[i][1] + self.treeplan.reward_sampled(self.ObservationValue[i])) * self.IntervalWeights[i]

		# Update reward
		# lower += self.mu - self.semi_tree.true_error
		# upper += self.mu + self.semi_tree.true_error
		r = self.treeplan.reward_analytical(self.mu, math.sqrt(self.semi_tree.variance))
		lower += r - self.semi_tree.true_error
		upper += r + self.semi_tree.true_error

		assert(lower <= upper), "Lower > Upper!, %s, %s" %(lower,upper)

		return lower, upper

	def UpdateChildrenBounds(self, index_updated):
		""" Update bounds of OTHER children while taking into account lipschitz constraints
		@param index_updated: index of child whose bound was just updated
		"""

		lip = self.semi_tree.lipchitz

		assert self.BoundsChildren[index_updated][0] <= self.BoundsChildren[index_updated][1], "%s, %s" % (self.BoundsChildren[index_updated][0],self.BoundsChildren[index_updated][1])
		# Intervals lying to the left of just updated interval
		for i in reversed(xrange(index_updated)):
			change = False
			testLower = self.BoundsChildren[i+1][0] - lip * (self.ObservationValue[i+1] - self.ObservationValue[i])
			testUpper = self.BoundsChildren[i+1][1] + lip * (self.ObservationValue[i+1] - self.ObservationValue[i])
			#print self.BoundsChildren[i], testLower, testUpper
			if self.BoundsChildren[i][0] < testLower:
				change = True
				self.BoundsChildren[i] = (testLower, self.BoundsChildren[i][1])

			if self.BoundsChildren[i][1] > testUpper:
				change = True
				self.BoundsChildren[i] = (self.BoundsChildren[i][0], testUpper)

			assert(self.BoundsChildren[i][0] <= self.BoundsChildren[i][1]), "lower bound greater than upper bound %f, %f" % (self.BoundsChildren[i][0], self.BoundsChildren[i][1])

			if not change == True:
				break

		# Intervals lying to the right of just updated interval
		for i in xrange(index_updated+1, len(self.ActionChildren)):
			change = False
			testLower = self.BoundsChildren[i-1][0] - lip * (self.ObservationValue[i] - self.ObservationValue[i-1])
			testUpper = self.BoundsChildren[i-1][1] + lip * (self.ObservationValue[i] - self.ObservationValue[i-1])
			if self.BoundsChildren[i][0] < testLower:
				change = True
				self.BoundsChildren[i] = (testLower, self.BoundsChildren[i][1])

			if self.BoundsChildren[i][1] > testUpper:
				change = True
				self.BoundsChildren[i] = (self.BoundsChildren[i][0], testUpper)

			assert(self.BoundsChildren[i][0] <= self.BoundsChildren[i][1]), "lower bound greater than upper bound %f, %f" % (self.BoundsChildren[i][0], self.BoundsChildren[i][1])

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

		target = int(math.floor(self.num_partitions/2))
		num_nodes_expanded += self.SkeletalExpandHere(target)
		self.UpdateChildrenBounds(target)
		return num_nodes_expanded

	def SkeletalExpandHere(self, index_to_expand):
		""" Expand given node at a particular index
		"""
		num_nodes_expanded = 0
		assert self.ActionChildren[index_to_expand] == None, "Node already expanded"
		self.ActionChildren[index_to_expand] = MCTSActionNode(TransitionH(self.augmented_state, self.ObservationValue[index_to_expand]), self.semi_tree, self.treeplan, self.lamb)
		num_nodes_expanded += self.ActionChildren[index_to_expand].SkeletalExpand()
		lower, upper = self.ActionChildren[index_to_expand].Eval()
		assert lower <= upper
		#print lower, upper, self.BoundsChildren[index_to_expand]
		self.BoundsChildren[index_to_expand] = (max(self.BoundsChildren[index_to_expand][0], lower), min(self.BoundsChildren[index_to_expand][1], upper))
		#print self.BoundsChildren[index_to_expand]
		assert self.BoundsChildren[index_to_expand][0] <= self.BoundsChildren[index_to_expand][1]
		self.UpdateChildrenBounds(index_to_expand)

		if self.ActionChildren[index_to_expand].saturated:
			self.numchild_unsaturated -= 1
			if self.numchild_unsaturated == 0: self.saturated = True

		return num_nodes_expanded



def MCTSExpand(self, epsilon, gamma, x_0, H, max_nodes = 10**15):
		print "Preprocessing weight spaces..."
		st = self.Preprocess(x_0.physical_state, x_0.history.locations[0:-1], H, epsilon)

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
		print bestval,best_a

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
			if (obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0])*obs_node.IntervalWeights[i] > highest_variance:
				most_uncertain_node_index = i
				highest_variance = (obs_node.BoundsChildren[i][1] - obs_node.BoundsChildren[i][0])*obs_node.IntervalWeights[i]

		i = most_uncertain_node_index
		# If observation is leaf, then we expand:
		if obs_node.ActionChildren[i] == None:

			new_action_node = MCTSActionNode(TransitionH(obs_node.augmented_state, obs_node.ObservationValue[i]), new_semi_tree, self, l)
			obs_node.ActionChildren[i] = new_action_node

			num_nodes_expanded = new_action_node.SkeletalExpand()
			# Update upper and lower bounds on this observation node
			lower, upper = new_action_node.Eval()
			obs_node.BoundsChildren[i] = (max(obs_node.BoundsChildren[i][0], lower), min(obs_node.BoundsChildren[i][1], upper))

		else: # Observation has already been made, expand further

			lower, upper, num_nodes_expanded = self.MCTSRollout(obs_node.ActionChildren[i], new_semi_tree, T-1, l)
			obs_node.BoundsChildren[i] = (max(lower, obs_node.BoundsChildren[i][0]), min(upper, obs_node.BoundsChildren[i][1]))

		obs_node.UpdateChildrenBounds(i)
		lower, upper = obs_node.Eval()
		assert(lower <= upper)
		action_node.BoundsChildren[best_a] = (max(action_node.BoundsChildren[best_a][0], lower), min(action_node.BoundsChildren[best_a][1], upper))

		if obs_node.ActionChildren[i].saturated:
			obs_node.numchild_unsaturated -= 1
			if obs_node.numchild_unsaturated == 0:
				obs_node.saturated = True

		#action_node.DetermineDominance()
		action_node.DetermineSaturation()

		return action_node.Eval() + (num_nodes_expanded, )