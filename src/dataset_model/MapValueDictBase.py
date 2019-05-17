import numpy as np


class MapValueDictBase():
    # needed for rounding while adding into dict
    ROUNDING_CONST = 5

    def __init__(self, locations, values):
        self.locations = locations
        self.values = values

        self.start_location = None

        self.__vals_dict = {}
        for i in range(self.locations.shape[0]):
            rounded_location = np.around(self.locations[i], decimals=self.ROUNDING_CONST)
            self.__vals_dict[tuple(rounded_location)] = self.values[i]

    def __call__(self, query_location):
        tuple_loc = (
            round(query_location[0], ndigits=self.ROUNDING_CONST),
            round(query_location[1], ndigits=self.ROUNDING_CONST))
        assert tuple_loc in self.__vals_dict, "No close enough match found for query location " + str(query_location)
        return self.__vals_dict[tuple_loc]

    def WriteToFile(self, filename):
        vals = np.atleast_2d(self.values).T
        concatenated_dataset = np.concatenate((self.locations, vals), axis=1)
        np.savetxt(filename, concatenated_dataset, fmt='%11.8f')
