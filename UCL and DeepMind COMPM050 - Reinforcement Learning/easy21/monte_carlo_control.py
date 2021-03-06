"""
The goal of this assignment is to apply reinforcement learning methods to a
simple card game that we call Easy21. This exercise is similar to the Blackjack
example in Sutton and Barto 5.3 – please note, however, that the rules of the
card game are different and non-standard.
"""
from collections import defaultdict

import numpy as np

from environment import step, draw_initial_state
from reinforcement_learning_algorithm import ReinforcementLearningAlgorithm
from viz import plot_v

# Q2: Monte-Carlo Control in Easy21
"""
Apply Monte-Carlo control to Easy21. Initialise the value function to zero.
Use a time-varying scalar step-size of αt = 1/N(st,at) and an ε-greedy
exploration strategy with εt = N0/(N0 + N(st)), where N0 = 100 is a constant,
N(s) is the number of times that state s has been visited, and N(s,a) is the
number of times that action a has been selected from state s. Feel free to
choose an alternative value for N0, if it helps producing better results. Plot
the optimal value function V ∗ (s) = maxa Q∗ (s, a) using similar axes to the
following figure taken from Sutton and Barto’s Blackjack example.
"""


class MonteCarloControl(ReinforcementLearningAlgorithm):
    """
    Algorithm for finding an optimal policy by using Monte Carlo
    control method
    """
    def __init__(self, N0, gamma):
        self.N0 = N0
        self.gamma = gamma
        self.N_s = defaultdict(int)  # Number of times state s has been visited
        self.N_sa = defaultdict(int)  # Number of times action a has been selected from state s
        self.q = defaultdict(int)  # Action-value function

    def run(self, n_episodes: int):
        for i in range(n_episodes):
            is_terminal = False
            episode_trajectory = list()
            s0 = draw_initial_state()
            while not is_terminal:
                a = self.pick_action_eps_greedy(s0)
                s, r, is_terminal = step(s0, a)
                episode_trajectory.append((s0, a, r))
                # R_sa[(s0, a)].append(r)
                self.N_sa[(s0, a)] += 1
                self.N_s[s0] += 1
                s0 = s
            G = 0  # Expected cumulative reward
            for sar in episode_trajectory[::-1]:
                s, a, r = sar
                G = self.gamma * G + r
                alpha = 1 / self.N_sa[(s, a)]
                self.q[(s, a)] += alpha * (G - self.q[(s, a)])


n_episodes = 1000000
mcmc = MonteCarloControl(N0=100, gamma=0.8)
mcmc.run(n_episodes)
v = mcmc.to_v()
plot_v(v)
