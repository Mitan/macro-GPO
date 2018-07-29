import numpy as np


def TupleToLine(tuple_location, dim_1, dim_2):
    float_line =  tuple_location[0] * dim_2 + tuple_location[1] + 1
    return int(float_line)


def LineToTuple(line_location, dim_1, dim_2):
    return (  float((line_location - 1) / dim_2),   float((line_location - 1) % dim_2)  )

# arguments are lists
def GenerateGridPairs(first_range, second_range):
    g = np.meshgrid(first_range, second_range)
    pairs = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    return pairs


# converts ndarray to tuple
# can't pass ndarray as a key for dict
def ToTuple(arr):
    return tuple(map(tuple, arr))


# generates a set of reachable locations for simulted agent
def generate_set_of_reachable_locations(start, b_size, gap):
        steps = 20 / b_size
        a = []
        s_x, s_y = start
        for st in range(steps):
            a = a + [(round(s_x + i * gap, 2), s_y + gap * st * b_size) for i in
                     range(-20 + st * b_size, 21 - st * b_size)]
            a = a + [(s_x + gap * st * b_size, round(s_y + i * gap, 2)) for i in
                     range(-20 + st * b_size, 21 - st * b_size)]
            a = a + [(s_x - gap * st * b_size, round(s_y + i * gap, 2)) for i in
                     range(-20 + st * b_size, 21 - st * b_size)]
            a = a + [(round(s_x + i * gap, 2), s_y - gap * st * b_size) for i in
                     range(-20 + st * b_size, 21 - st * b_size)]

        return list(set(a))


if __name__ == '__main__':
    b = 4
    g = 0.05
    s = (1.0, 1.0)
    answer = generate_set_of_reachable_locations(b_size=b, gap=g, start=s)

    x = [t[0] for t in answer]
    y = [t[1] for t in answer]
    import matplotlib.pyplot as plt

    plt.scatter(x, y)
    plt.show()