def run_experiment(env, agent, episodes=100):
    successful_rewards = []
    rewards = []
    reflections = []

    best_reward = float('-inf')
    best_path = None

    for ep in range(episodes):
        # print(f"\nEpisode {ep + 1}: ", end="")
        r, refl, path, is_successful = agent.train_episode()
        if is_successful:
            successful_rewards.append(r)
        rewards.append(r)
        reflections.append(refl)

        if r > best_reward and is_successful:
            best_reward = r
            best_path = path

        if refl > 0:
            # print(f"Episode {ep + 1} | Reward = {r:.2f} | Reflections Used = {refl}")
            pass

    if best_path is not None:
        print(f"Success! Best Reward: {best_reward:.2f}")
        env.plot_path(best_path)
    else:
        print("Failed.")
    return successful_rewards, rewards, reflections