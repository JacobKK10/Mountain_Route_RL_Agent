from mountain_env import MountainRouteEnv
from car_agent import RandomAgent

env = MountainRouteEnv(render_mode="human")
agent = RandomAgent(env.act_space)

n_episodes = 10

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        env.render()
        total_reward += reward

    print(f"Episode {episode+1} reward = {total_reward:.2f}")

env.close()
