

# TODO note history includes current state
# just state and history
class AugmentedState:
    def __init__(self, physical_state, initial_history):
        """
        Initialize augmented state with initial position and history
        """
        # 2D array
        self.physical_state = physical_state
        # 2D array
        self.history = initial_history

    def to_str(self):
        return \
            "Physical State\n" + \
            str(self.physical_state) + "\n" + \
            "Locations\n" + \
            str(self.history.locations) + "\n" + \
            "Measurements\n" + \
            str(self.history.measurements)