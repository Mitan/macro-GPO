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
               MyopicUCB=5,
               BUCB_PE=6,
               new_qEI=7,
               BUCB=8,
               LP=9)
