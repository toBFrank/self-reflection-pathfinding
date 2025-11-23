def run_experiment(env, agent, episodes=100):
    rewards = []
    for ep in range(episodes):
        r, refl = agent.train_episode()
        rewards.append(r)

        if refl > 0:
            print(f"Episode {ep} | Reward = {r:.2f} | Reflections Used = {refl}")

    return rewards