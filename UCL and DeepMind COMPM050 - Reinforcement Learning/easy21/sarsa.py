from collections import defaultdict

import numpy as np

from environment import draw_initial_state, step
from reinforcement_learning_algorithm import ReinforcementLearningAlgorithm
from viz import plot_v


class Sarsa(ReinforcementLearningAlgorithm):
    def __init__(self, N0, lambda_, alpha):
        # lambda_ here == lambda_ (Sutton&Barto)
        self.N0 = N0
        self.N_s = defaultdict(int)
        self.N_sa = defaultdict(int)
        # Eligibility trace
        self.E = defaultdict(int)
        self.lambda_ = lambda_
        self.alpha = alpha
        # Action-value function
        self.q = defaultdict(int)

    def run(self, n_episodes: int):
        # Most likely ned to convert dicts to tensors and update mulitple values
        #  at the same time for a given episode
        for i in range(n_episodes):
            is_terminal = False
            s = draw_initial_state()
            a = self.pick_action_eps_greedy(s)
            while not is_terminal:
                self.N_s[s] += 1
                self.N_sa[(s, a)] += 1
                s2, r, is_terminal = step(s, a)
                if not is_terminal:
                    a2 = self.pick_action_eps_greedy(s2)
                    delta = r + self.q[(s2, a2)] - self.q[(s, a)]
                else:
                    delta = r - self.q[(s, a)]
                self.E[(s, a)] += 1
                self.q[(s, a)] += self.alpha * delta * self.E[(s, a)]
                for k in self.E.keys():
                    self.E[k] *= self.lambda_
                s = s2


n_episodes = 100000
sarsa = Sarsa(N0=100, lambda_=.9, alpha=0.1)
sarsa.run(n_episodes)
v = sarsa.to_v()
plot_v(v)
