

def GetNumberOfSamples(horizon, t):
    if horizon == 4:
        return __Sample_funciton_4(t)
    elif horizon == 3:
        return __Sample_funciton_3(t)
    elif horizon == 2:
        return __Sample_funciton_2(t)
    elif horizon == 1:
        return __Sample_funciton_1(t)
    else:
        raise Exception("Unimplemented horizon")


def __Sample_funciton_4(t):
    if t == 4:
        return 150
    elif t == 3:
        return 100
    elif t == 2:
        return 50
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return -1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_3(t):
    if t == 3:
        return 500
    elif t == 2:
        return 100
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return -1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_2(t):
    if t == 2:
        return 100
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return -1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_1(t):
    # Myopic case, sampling not needed
    if t == 1:
        return -1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

if __name__ == "__main__":
    H = 4
    t = 2
    print GetNumberOfSamples(H, t)