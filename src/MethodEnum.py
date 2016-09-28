from enum import Enum


class MethodEnum(Enum):
    Exact = 1
    Anytime = 2
    qEI = 3
    MLE = 4
    MyopicUCB = 5
