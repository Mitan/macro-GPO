from src.enum.DatasetEnum import DatasetEnum


def get_domain_descriptor(dataset_type):
    if dataset_type == DatasetEnum.Robot:
        return RobotDomainDescriptor()
    elif dataset_type == DatasetEnum.Road:
        return RoadDomainDescriptor()
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
