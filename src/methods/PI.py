import math
from scipy.stats import norm


def method_PI(x_0, gp, next_states):
    best_observation = max(x_0.history.measurements)

    # next_states = self.GetNextAugmentedStates(x_0)

    vBest = 0.0
    xBest = next_states[0]

    current_locations = x_0.history.locations
    current_chol = gp.Cholesky(x_0.history.locations)

    for x_next in next_states:

        next_physical_state = x_next.physical_state
        var = gp.GPVariance(locations=current_locations, current_location=next_physical_state,
                            cholesky=current_chol)
        sigma = math.sqrt(var[0, 0])

        weights = gp.GPWeights(locations=current_locations, current_location=next_physical_state,
                               cholesky=current_chol)
        mu = gp.GPMean(measurements=x_0.history.measurements, weights=weights)[0]

        Z = (mu - best_observation) / sigma

        probImprovement = norm.cdf(x=Z, loc=0, scale=1.0)
        # probImprovement = 1.0 - norm.cdf(x=best_observation, loc=mu, scale=sigma)

        if probImprovement >= vBest:
            vBest = probImprovement
            xBest = x_next

    return vBest, xBest, -1
