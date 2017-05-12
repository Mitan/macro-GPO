import numpy as np


input_file_name = '../datasets/intel-robot/coordinates.txt'
all_points = np.genfromtxt(input_file_name)

neighbours_dict = []

num_points, _ = all_points.shape

treshhold = 9
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