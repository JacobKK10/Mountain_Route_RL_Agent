from mountain_route_choice_env import MountainRouteChoiceEnv
from route_choice_agent import CarDecisionAgent
import matplotlib.pyplot as plt

env = MountainRouteChoiceEnv(render_mode=None)
agent = CarDecisionAgent(env.action_space, alpha=0.1, gamma=0.95, epsilon=0.1)

n_episodes = 500
rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    agent.update(state, action, reward, next_state)
    rewards.append(reward)

    if episode % 50 == 0:
        print(f"Episode {episode}, reward: {reward:.2f}")

env.close()

plt.figure(figsize=(8,4))
plt.plot(rewards, marker=".", alpha=0.7)
plt.title("Agent training progress (route choice)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

env = MountainRouteChoiceEnv(render_mode="human")
for i in range(5):
    state, _ = env.reset()
    action = agent.select_action(state)
    next_state, reward, done, _, _ = env.step(action)
    env.render()
    print(f"Test {i+1}: route {'A' if action == 0 else 'B'}, reward: {reward:.2f}")
    input("Press Enter...")
env.close()