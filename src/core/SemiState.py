# updated
class SemiState:
    """ State which only contains locations visited and its current location
    """

    def __init__(self, physical_state, locations):
        self.physical_state = physical_state
        # hsitory doesn't include current batch
        # it is more convenient since we do not know mean and var in current batch
        # and locations is history for predicting it
        self.locations = locations