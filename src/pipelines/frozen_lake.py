import gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1")  # <- v1 works in Gym 0.20.0
state = env.reset()

num_episodes = 5
max_steps = 100

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    print(f"\nEpisode {episode + 1}")

    for step in range(max_steps):
        # Take a random action (0=Left, 1=Down, 2=Right, 3=Up)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"Step {step + 1}: State={next_state}, Action={action}, Reward={reward}")

        state = next_state
        if done:
            print(f"Episode finished after {step + 1} steps with total reward {total_reward}")
            break

env.close()
print("\nSimulation finished!")