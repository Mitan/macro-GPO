import numpy as np
import copy


def TransitionP(augmented_state, action):
    """
        @return - copy of augmented state with physical_state updated
        """
    new_augmented_state = copy.deepcopy(augmented_state)
    # new macroaction
    new_augmented_state.physical_state = PhysicalTransition(new_augmented_state.physical_state, action)
    return new_augmented_state


def TransitionH(augmented_state, measurements):
    """
        @return - copy of augmented state with history updated
        """
    new_augmented_state = copy.deepcopy(augmented_state)
    # add new batch and measurements
    new_augmented_state.history.append(new_augmented_state.physical_state, measurements)
    return new_augmented_state


def PhysicalTransition(physical_state, macroaction):
    current_location = physical_state[-1, :]
    batch_size = macroaction.shape[0]

    repeated_location = np.asarray([current_location for i in range(batch_size)])
    # repeated_location = np.tile(current_location, batch_size)

    assert repeated_location.shape == macroaction.shape
    # new physical state is a batch starting from the current location (the last element of batch)
    new_physical_state = np.add(repeated_location, macroaction)

    # check that it is 2d
    assert new_physical_state.ndim == 2
    return new_physical_state