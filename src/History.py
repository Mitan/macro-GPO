import numpy as np


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
