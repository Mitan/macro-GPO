# from enum import Enum

"""
class MethodEnum(Enum):
    Exact = 1
    Anytime = 2
    qEI = 3
    MLE = 4
    MyopicUCB = 5
"""


def enum(**enums):
    return type('Enum', (), enums)


PlottingMethods = enum(TotalReward=1,
                       SimpleRegret=2,
                       Nodes=3,
                       TotalRewardBeta=4,
                       SimpleRegretBeta=5)