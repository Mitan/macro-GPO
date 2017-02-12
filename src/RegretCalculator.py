from StringIO import StringIO
import numpy as np


def CalculateMethodRegret(model_max, root_folder, method_name, batch_size):

    n_steps = 20 / batch_size
    regrets = []
    for i in range(n_steps):
        step_file_name = root_folder + method_name + '/step' + str(i) + '.txt'
        lines = open(step_file_name).readlines()
        first_line_index = 1 + batch_size + 1 + (1 + batch_size * (i+1)) + 1
        last_line_index = -1
        stripped_lines = map(lambda x: x.strip(), lines[first_line_index: last_line_index])
        joined_lines = " ".join(stripped_lines)
        assert joined_lines[0] == '['
        assert joined_lines[-1] == ']'
        a = StringIO(joined_lines[1:-1])

        # all measurements obtained by the robot till that step
        measurements = np.genfromtxt(a)

        assert measurements.shape[0] == 1 + batch_size * (i +1)
        # assert we parsed them all as numbers
        assert not np.isnan(measurements).any()

        max_found = max(measurements)
        regrets.append(model_max - max_found)
    return regrets


if __name__ == "__main__":
    # cannot use - cylcic linking
    folder_name = '../testsRoad/b5/18/seed0/'
    b = 5
    max_model = 0
    method  = 'anytime_h2'
    print CalculateMethodRegret(max_model, folder_name,method, b)
