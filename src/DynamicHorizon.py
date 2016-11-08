# todo change into class


def DynamicHorizon(t, H_max,  t_max):
    """
    :param t: current timestep
    :param H_max: maximum allowed horizon
    :param t_max: maximum number of timesteps
    :return:
    """
    return min(t_max - t, H_max)

if __name__=="__main__":
    T = 4
    H = 5
    for i in range(T):
        print DynamicHorizon(i, H, T)
