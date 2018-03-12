from src.Utils import ToTuple


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