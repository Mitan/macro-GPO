import numpy as np
from random import choice


class MapValueDictBase():
    # needed for rounding while adding into dict
    ROUNDING_CONST = 2

    def __init__(self, locations, values, epsilon=None):
        """
        @param epsilon - minimum tolerance level to determine equivalence between two points
        """

        self.locations = locations
        # the original mean of the values
        self.mean = np.mean(values)

        # self.values = values - self.mean
        self.values = values

        self.start_location = None

        """
        if not epsilon == None:
            self.epsilon = epsilon
            return

        ndims = locations.shape[1]
        self.epsilon = np.zeros((ndims,))
        for dim in xrange(ndims):
            temp = list(set(np.squeeze(locations[:, dim]).tolist()))
            temp = sorted(temp)
            self.epsilon[dim] = (min([temp[i] - temp[i - 1] for i in xrange(1, len(temp))])) / 4
        """
        self.__vals_dict = {}
        for i in range(self.locations.shape[0]):
            rounded_location = np.around(self.locations[i], decimals=self.ROUNDING_CONST)
            self.__vals_dict[tuple(rounded_location)] = self.values[i]

        # locations available as start point

        #self.StartLocations = list(self.locations)
        # print self.__vals_dict

    """
    def GenerateStartLocation(self, batch_size):
        return choice(list(self.locations))
    """
    def __call__(self, query_location):
        """
        Search for nearest grid point iteratively. Uses L1 norm as the distance metric

        bi = -1
        bd = None
        for i in xrange(self.locations.shape[0]):
            d = np.absolute(np.atleast_2d(query_location) - self.locations[i, :])
            l1 = np.sum(d)
            if np.all(d <= self.epsilon) and (bd == None or l1 < bd):
                bd = l1
                bi = i

        assert bd is not None, "No close enough match found for query location " + str(query_location)

        return self.values[bi]
        """

        tuple_loc = (
        round(query_location[0], ndigits=self.ROUNDING_CONST), round(query_location[1], ndigits=self.ROUNDING_CONST))
        assert tuple_loc in self.__vals_dict, "No close enough match found for query location " + str(query_location)
        return self.__vals_dict[tuple_loc]

    def WriteToFile(self, filename):
        vals = np.atleast_2d(self.values).T
        concatenated_dataset = np.concatenate((self.locations, vals), axis=1)
        np.savetxt(filename, concatenated_dataset, fmt='%11.8f')

    def GetMax(self):
        return max(self.values)