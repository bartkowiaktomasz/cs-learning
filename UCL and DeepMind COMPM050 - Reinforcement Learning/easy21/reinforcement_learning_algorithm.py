import numpy as np


class ReinforcementLearningAlgorithm:
    def pick_action_eps_greedy(self, s):
        eps = self.N0 / (self.N0 + self.N_s[s])
        if np.random.random_sample(1) > eps:
            if self.q[(s, 'stick')] > self.q[(s, 'hit')]:
                a = 'stick'
            elif self.q[(s, 'stick')] < self.q[(s, 'hit')]:
                a = 'hit'
            else:
                a = np.random.choice(['stick', 'hit'], p=[0.5, 0.5])
        else:
            a = np.random.choice(['stick', 'hit'], p=[0.5, 0.5])
        return a

    def to_v(self):
        v = dict()
        q = dict(self.q)
        for sa in q.keys():
            s = sa[0]
            v[s] = max(
                self.q[(s, 'stick')],
                self.q[(s, 'hit')]
            )
        return v
