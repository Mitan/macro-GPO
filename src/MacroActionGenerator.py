import numpy as np


def __GetStraightLineMacroAction(direction, length):
    return np.asarray([[direction[0] * i, direction[1] * i] for i in range(1, length+1)])


# Generates simple macroactions allowing to move straight in specified directions
def GenerateSimpleMacroactions(batch_size, grid_gap):
    action_set = ((0, grid_gap), (0, -grid_gap), (grid_gap, 0),(-grid_gap, 0))
    return [__GetStraightLineMacroAction(direction, batch_size) for direction in action_set]

if __name__ == "__main__":
    grid_gap = 1.0
    b_size = 3
    print GenerateSimpleMacroactions(b_size)