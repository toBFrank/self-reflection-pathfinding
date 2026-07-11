def run_experiment(env, agent, episodes=100):
    successful_rewards, rewards, reflections, alphas, epsilons, _, _, _ = run_experiment_with_paths(
        env, agent, episodes=episodes
    )
    return successful_rewards, rewards, reflections, alphas, epsilons


def run_experiment_with_paths(env, agent, episodes=100):
    successful_rewards = []
    rewards = []
    reflections = []
    episode_paths = []
    successful_episode_paths = []

    best_reward = float('-inf')
    best_path = None

    for ep in range(episodes):
        r, refl, path, is_successful, alphas, epsilons = agent.train_episode()
        if is_successful:
            successful_rewards.append(r)
            successful_episode_paths.append(path)
        rewards.append(r)
        reflections.append(refl)
        episode_paths.append(path)

        if r > best_reward and is_successful:
            best_reward = r
            best_path = path

    if best_path is not None:
        print(f"Success! Best Reward: {best_reward:.2f}")
    else:
        print("Failed.")
    return successful_rewards, rewards, reflections, alphas, epsilons, episode_paths, best_path, successful_episode_paths