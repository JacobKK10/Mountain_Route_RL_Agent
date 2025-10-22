import numpy as np

class CarDecisionAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def discretize(self, state):
        return tuple(np.round(state / 2).astype(int))

    def select_action(self, state):
        s = self.discretize(state)
        if s not in self.Q:
            self.Q[s] = np.zeros(self.action_space.n)
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[s])

    def update(self, state, action, reward, next_state):
        s = self.discretize(state)
        ns = self.discretize(next_state)
        if ns not in self.Q:
            self.Q[ns] = np.zeros(self.action_space.n)

        self.Q[s][action] += self.alpha * (reward + self.gamma * np.max(self.Q[ns]) - self.Q[s][action])