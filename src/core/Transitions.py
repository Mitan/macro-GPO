import copy


# transition of state to new_physical_state
def TransitionP(augmented_state, new_physical_state):
    """
        @return - copy of augmented state with physical_state updated
        """
    new_augmented_state = copy.deepcopy(augmented_state)
    # new macroaction
    new_augmented_state.physical_state = new_physical_state
    # new_augmented_state.physical_state = PhysicalTransition(new_augmented_state.physical_state, action)
    return new_augmented_state


def TransitionH(augmented_state, measurements):
    """
        @return - copy of augmented state with history updated
        """
    new_augmented_state = copy.deepcopy(augmented_state)
    # add new batch and measurements
    new_augmented_state.history.append(new_augmented_state.physical_state, measurements)
    return new_augmented_state
