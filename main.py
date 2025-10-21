from mountain_env import MountainRouteEnv
# from car_agent import RandomAgent
from car_agent import QlearningAgent
import matplotlib.pyplot as plt
import numpy as np

env = MountainRouteEnv(render_mode=None)
# agent = RandomAgent(env.act_space)
agent = QlearningAgent(env.act_space)


n_episodes = 5000

episode_rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        # state, reward, done, _, _ = env.step(action)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        # env.render()
        state = next_state
        total_reward += reward
        
    episode_rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{n_episodes} | Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}")

env.close()

window = 50
smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,6))
plt.plot(episode_rewards, marker="o")
plt.plot(smoothed_rewards, color='red')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Agent training progress")
plt.grid(True)
plt.show()