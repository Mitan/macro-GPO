__author__ = 'a0134673'
import numpy as np

"""
 class to store dataset
 note assume that locations are np arrays or lists
"""
class MapDatasetStorage():
    # not efficient
    # not ensures that all point are different
    def __init__(self, locations, measurements):
        self.dictionary = {}
        number_of_points = locations.shape[0]
        assert number_of_points == measurements.shape[0]
        for i in range(number_of_points):
            tuple_key = self.ToTuple(locations[i])
            self.dictionary[tuple_key] = measurements[i]

    def __call__(self, query_location):
        tuple_key = self.ToTuple(query_location)

        if tuple_key in self.dictionary:
            return self.dictionary[tuple_key]
        else:
            raise KeyError("Point " + str(tuple_key)+ " is not in dataset")

    def ToTuple(self, arr):
        #if isinstance(arr, list):
        #return tuple(map(tuple, arr))
        return tuple(arr)

if __name__ == "__main__":

    file = open("./datasets/bball.dat")
    data = np.genfromtxt(file,skip_header=10)
    file.close()

    # restrict the field
    indexes_x = [i for i in range(data.shape[0]) if data[i,0] > 4  and data[i,1] < 19 and data[i,1] > 6]
    # restricted full data
    data = data[indexes_x, :]
    number_of_points = data.shape[0]
    #print number_of_points

    X_values = data[:, 0:2]
    #print X_values[0]
    #print max(data[:, 0:1])
    #K_normal = data[:, 2:3]
    K_log = data[:, 3]
    #P_normal = data[:, 5:6]
    P_log = data[:, 6]

    K_dataset = MapDatasetStorage(X_values, K_log)
    P_dataset = MapDatasetStorage(X_values, P_log)
    for i in range(number_of_points):
        pass
        #print K_dataset(X_values[i])

    bad_places = [[5., 8.], [18., 17.]]
    print P_dataset([18, 17])