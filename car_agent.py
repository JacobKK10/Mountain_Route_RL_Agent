import numpy as np

#For basic testing
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, state):
        return self.action_space.sample()
    
class QlearningAgent:
    def __init__(self, action_space, n_bins=20, alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.05):
        self.action_space = action_space
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    
        self.position_bins = np.linspace(0, 1, n_bins)
        self.velocity_bins = np.linspace(-0.05, 0.05, n_bins)

        self.Q = np.zeros((n_bins, n_bins, action_space.n))

    def discretize_state(self, state):
        position, velocity = state
        pos_idx = np.digitize(position, self.position_bins) - 1
        vel_idx = np.digitize(velocity, self.velocity_bins) - 1
        pos_idx = np.clip(pos_idx, 0, self.n_bins - 1)
        vel_idx = np.clip(vel_idx, 0, self.n_bins - 1)
        return pos_idx, vel_idx

    def select_action(self, state):
        pos_idx, vel_idx = self.discretize_state(state)

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[pos_idx, vel_idx])

    def update(self, state, action, reward, next_state, done):
        pos_idx, vel_idx = self.discretize_state(state)
        next_pos_idx, next_vel_idx = self.discretize_state(next_state)

        best_next_action = np.argmax(self.Q[next_pos_idx, next_vel_idx])
        td_target = reward + self.gamma * self.Q[next_pos_idx, next_vel_idx, best_next_action] * (1 - done)
        td_error = td_target - self.Q[pos_idx, vel_idx, action]
        self.Q[pos_idx, vel_idx, action] += self.alpha * td_error

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)