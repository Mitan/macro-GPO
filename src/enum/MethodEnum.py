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
               BucbPE=6,
               EI=7,
               PI=8,
               new_qEI=9,
               BUCB=10,
               LP=11,
               Rollout=12)
