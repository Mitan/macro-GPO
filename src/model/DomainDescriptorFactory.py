from src.enum.DatasetEnum import DatasetEnum


def get_domain_descriptor(dataset_type):
    if dataset_type == DatasetEnum.Robot:
        return RobotDomainDescriptor()
    elif dataset_type == DatasetEnum.Road:
        return RoadDomainDescriptor()
    elif dataset_type == DatasetEnum.Simulated:
        return SimulatedDomainDescriptor()
    else:
        raise ValueError("Unknown dataset")


class RobotDomainDescriptor:
    def __init__(self):
        # domain is not grid-like
        self.grid_gap = None

        self.grid_domain = None

        self.domain_size = 145


class RoadDomainDescriptor:
    def __init__(self):

        self.grid_gap = 1.0

        # upper values are not included
        self.grid_domain = ((0.0, 50.0), (0.0, 100.0))

        self.domain_size = 50 * 100


class SimulatedDomainDescriptor:

    def __init__(self):

        self.grid_gap = 0.05

        # unused
        # number of samples in each dimension
        self.num_samples_grid = (50, 50)

        # upper values are not included
        self.grid_domain = ((-0.25, 2.25), (-0.25, 2.25))
        self.domain_size = 50 * 50
