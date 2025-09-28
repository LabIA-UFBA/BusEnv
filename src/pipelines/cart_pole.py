import gym

env = gym.make("CartPole-v1")
state = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()

env.close()
print("Simulation finished (headless, no window).")