from numpy import random

from DeterministicGPP import DeterministicEpsGPP
from utils import GetMeasurement
from ExperimentSetup import R

__author__ = 'Dmitrii'

# time of experiment
t = 20
numberOfSamples = 5

eps = 0.1
# since n = 20, grid is [-10, 10]
# initially robot is in the center
s = [0, 0]
z = random.random()
d = [[s, z]]
rewards = []

for i in range(t):
    s_next = DeterministicEpsGPP(d, eps, numberOfSamples)
    z_next = GetMeasurement(s, d)
    d_next = [s_next, z_next]
    rewards.append(R(z_next, s_next))
    d.append(d_next)