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


Methods = enum(Exact=1,
               Anytime=2,
               qEI=3,
               MLE=4,
               MyopicUCB=5)
