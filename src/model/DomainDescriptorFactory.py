from src.enum.DatasetEnum import DatasetEnum


def get_domain_descriptor(dataset_type):
    if dataset_type == DatasetEnum.Robot:
        return RobotDomainDescriptor()
    else:
        raise ValueError("Unknown dataset")


class RobotDomainDescriptor:
    def __init__(self):
        # domain is not grid-like
        self.grid_gap = None

        self.grid_domain = None

        self.domain_size = 145
