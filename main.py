from mountain_env import MountainRouteEnv
from car_agent import RandomAgent
import matplotlib.pyplot as plt

env = MountainRouteEnv(render_mode=None)
agent = RandomAgent(env.act_space)

n_episodes = 100

episode_rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        env.render()
        total_reward += reward
        
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1} reward = {total_reward:.2f}")

env.close()

plt.figure(figsize=(10,6))
plt.plot(episode_rewards, marker="o")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Agent training progress")
plt.grid(True)
plt.show()
