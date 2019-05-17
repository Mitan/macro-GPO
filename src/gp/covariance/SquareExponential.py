import numpy as np


class SquareExponential:
    def __init__(self, length_scale, signal_variance):
        """
        @param: length_scale l - array or list containing the length scales for each dimension
        @param: signal variance sigma_f_squared - float containing the signal variance
        """

        self.length_scale = np.atleast_2d(length_scale)
        self.signal_variance = signal_variance

    def Cov(self, physical_state_1, physical_state_2):
        diff = np.atleast_2d(physical_state_1) - np.atleast_2d(physical_state_2)
        length_scale_squared = np.square(self.length_scale)
        squared = np.dot(diff, np.divide(diff, length_scale_squared).T)
        return self.signal_variance * np.exp(-0.5 * squared)
