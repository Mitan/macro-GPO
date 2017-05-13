import numpy as np


def GenerateUnconstrainedNeighbours(treshhold):
    input_file_name = '../datasets/intel-robot/coordinates.txt'
    all_points = np.genfromtxt(input_file_name)

    neighbours_dict = []

    num_points, _ = all_points.shape

    # treshhold = 9
    for i in range(num_points):
        id = all_points[i, 0]
        coord = all_points[i, 1:]
        neighbours = [id]

        for j in range(num_points):
            if j == i:
                continue
            other_id = all_points[j, 0]
            other_coord = all_points[j, 1:]
            if np.linalg.norm(coord - other_coord) < treshhold:
                neighbours.append(other_id)


        neighbours_dict.append(neighbours)


    output_filename = '../datasets/intel-robot/neighbours_raw.txt'
    output_file = open(output_filename, 'w')

    for id in neighbours_dict:
        str_list = map(str, id)
        joined_list = " ".join(str_list)
        output_file.write(joined_list + '\n')

    output_file.close()

# nodes are indexed from 1
def CheckNeighboursConsistency():
    input_file = 'neighbours.txt'
    all_lines = open(input_file).readlines()
    int_all_lines = map(lambda l: map(float, l.split()), all_lines)
    for l in int_all_lines:
        id = l[0]
        neighbours = l[1:]
        for n_id in neighbours:
            # nodes are indexed from 1
            n_neighbours = int_all_lines[int(n_id)-1]
            assert n_neighbours[0] == n_id
            assert id in n_neighbours[1:], "%r == %r" % (id, n_id)

def AddFakeNodesAndCoords():
    current_node_id = 55.0

    input_neighbours_file = 'neighbours.txt'
    all_neighbours_lines = open(input_neighbours_file).readlines()
    # list of lines
    int_neighbours_lines = map(lambda l: map(float, l.split()), all_neighbours_lines)

    input_file_name = '../datasets/intel-robot/coordinates.txt'
    all_coord_points = np.genfromtxt(input_file_name)

    fake_neighbours_file = open('fake_neighbours.txt', 'w')
    fake_coords_file = open('fake_coordinates.txt', 'w')

    for l in int_neighbours_lines:
        id = l[0]
        neighbours = l[1:]
        for n in neighbours:
            if id < n:
                fake_neighbours_file.write(str(current_node_id)+ ' ' +  str(id) + ' ' + str(n) + '\n')

                id_coords = all_coord_points[int(id) - 1, 1:]
                assert all_coord_points[int(id)-1, 0] == id

                n_coords = all_coord_points[int(n) - 1, 1:]
                assert all_coord_points[int(n) - 1, 0] == n
                fake_coords = (n_coords + id_coords) / 2

                fake_coords_file.write(str(current_node_id) + ' ' + str(fake_coords[0]) + ' ' + str(fake_coords[1]) + '\n')

                current_node_id+=1

    fake_neighbours_file.close()
    fake_coords_file.close()




AddFakeNodesAndCoords()
# CheckNeighboursConsistency()
# GenerateUnconstrainedNeighbours(9)
