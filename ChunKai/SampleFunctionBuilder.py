def GetSampleFunction(horizon, t):
    if horizon ==5:
        return __Sample_funciton_5(t)
    elif horizon == 4:
        return __Sample_funciton_4(t)
    elif horizon == 3:
        return __Sample_funciton_3(t)
    elif horizon == 2:
        return __Sample_funciton_2(t)
    elif horizon == 1:
        return __Sample_funciton_1(t)
    else:
        raise Exception("Unimplemented horizon")

def __Sample_funciton_5(t):
    if t == 5:
        return 5
    elif t == 4:
        return 4
    elif t == 3:
        return 3
    # case of MLE, should be held by upper lvl
    elif t == 2:
        return 1
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return 1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_4(t):
    if t == 4:
        return 5
    elif t == 3:
        return 4
    elif t == 2:
        return 4
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return 1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_3(t):
    if t == 3:
        return 5
    elif t == 2:
        return 2
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return 1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_2(t):
    if t == 2:
        return 5
    # case of myopic, no need to sample, should be held by upper lvl
    elif t == 1:
        return 1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

def __Sample_funciton_1(t):
    # Myopic case, sampling not needed
    if t == 1:
        return 1
    else:
        raise ValueError("wrong value of lvl - out of range horizon")

if __name__ == "__main__":
    H = 4
    test_f_1 = lambda t : GetSampleFunction(H,t)
    print test_f_1(5)