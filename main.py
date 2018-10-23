
import numpy as np
from algorithm.epsilon_greedy import EpsilonGreedy
from algorithm.decreasing_epsilon_greedy import DecreasingEpsilonGreedy
from algorithm.optimistic_initial_value import OptimisticInitialValue
from algorithm.ucb import UCB1
from algorithm.bayes_thompson_sampling import BayesThompsonSampling

import copy
from time import sleep

import matplotlib.pyplot as plt

N = 100000 # simulation times
epsilon = 0.1

result = EpsilonGreedy.run(0.8, 0.5, 0.2, epsilon, N) #mu1,mu2,m3,epsilon
plt.plot(result)

result = DecreasingEpsilonGreedy.run(0.8, 0.5, 0.2, epsilon, N) #mu1,mu2,m3,epsilon
plt.plot(result)

result = OptimisticInitialValue.run(0.8, 0.5, 0.2, epsilon, N) #mu1,mu2,m3,epsilon
plt.plot(result)

result = UCB1.run(0.8, 0.5, 0.2, epsilon, N) #mu1,mu2,m3,epsilon
plt.plot(result)

result = BayesThompsonSampling.run(0.8, 0.5, 0.2, epsilon, N) #mu1,mu2,m3,epsilon
plt.plot(result)

labels = ['Epsilon greedy',
          'DecreasingEpsilonGreedy',
          'OptimisticInitialValue',
          'UCB',
          'BayesThompsonSampling']
plt.legend(labels, loc='lower right')
plt.xscale('log')
plt.show()