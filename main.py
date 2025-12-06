#  main.py
import os
import numpy as np
from experiment.gridworld import GridWorld
from agents.q_agent import QAgent
from agents.reflection.adaptive_reflection import AdaptiveReflection
from agents.reflection.epsilon_reflection import EpsilonReflection
from agents.reflection.reflection_costs import ReflectionCost
from experiment.runner import run_experiment

from analysis.plotting import (
    plot_alpha_over_time,
    plot_epsilon_over_time,
    plot_reward_over_time,
    plot_successful_rewards_over_time,
    plot_reflections_over_time,
    plot_success_counts,
)

from analysis.tables import (
    write_summary_md,
    )

# -------------------------------------------------
# Assess difficulty of environment
# -------------------------------------------------
def assess_environment_difficulty(env, num_runs=3, episodes=300):
    # run baseline agent on environment 10 times and get average steps to goal
    base_agent = QAgent(env, use_reflection=False)
    successful_episodes = 0

    for _ in range(num_runs):
        succ, rewards, refl, alphas, epsilons = run_experiment(env, base_agent, episodes=episodes)
        successful_episodes += len(succ)

    avg_success_rate = successful_episodes / (num_runs * episodes)
    if avg_success_rate > 0.95:
        difficulty = "EASY"
    elif avg_success_rate > 0.25:
        difficulty = "MEDIUM"
    else:
        difficulty = "HARD"
    print(f"[INFO] Environment assessed as {difficulty} (Success Rate: {avg_success_rate*100:.2f}%)")
    return difficulty, avg_success_rate



# -------------------------------------------------
# Run a single environment
# -------------------------------------------------
def run_agents_on_env(env):
    results = []

    # Baseline
    print("\n=== Running Agent: Baseline ===")
    base_agent = QAgent(env, use_reflection=False)
    base_succ, base_rewards, base_refl, base_alphas, base_epsilons = run_experiment(env, base_agent, episodes=300)
    results.append(("Baseline", base_succ, base_rewards, base_refl, base_alphas, base_epsilons))

    # Adaptive low / medium / high
    for cost, name, color in [
        (ReflectionCost.HIGH.value, "Reflection High Cost", "purple"),
        (ReflectionCost.MEDIUM.value, "Reflection Medium Cost", "red"),
        (ReflectionCost.LOW.value, "Reflection Low Cost", "green"),
    ]:
        print(f"\n=== Running Agent: {name} ===")
        env.set_reflect_cost(cost)
        agent = QAgent(env, use_reflection=True, reflection_strategy=AdaptiveReflection())
        succ, rewards, refl, alphas, epsilons = run_experiment(env, agent, episodes=300)
        results.append((name, succ, rewards, refl, alphas, epsilons))

    return results

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":
    # Base environment
    env = GridWorld(size=20, min_distance=15)
    env.generate_environment(num_obstacles=150)

    print("\n=== Assessing Environment Difficulty ===")
    difficulty, avg_success_rate = assess_environment_difficulty(env)
    print("=====================================\n")

    # Specify where to store results
    run_name = input(f"Enter a name for this {difficulty} run (e.g., run_1): ").strip()

    save_dir = os.path.join("results", run_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Results will be saved to: {save_dir}")

    results = run_agents_on_env(env)

    # Produce reward-over-time plot
    plot_reward_over_time(
        [(name, rewards, color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "reward_over_time.png")
    )
    
    # Produce successful rewards over time plot
    plot_successful_rewards_over_time(
        [(name, succ, color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "successful_rewards_over_time.png")
    )

    # Produce reflection-over-time plot
    plot_reflections_over_time(
        [(name, refl, color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "reflections_over_time.png")
    )

    # Success counts
    plot_success_counts(
        [(name, len(succ), color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "success_counts.png")
    )

    # Alpha and Epsilon over time plot
    plot_alpha_over_time(
        [(name, alphas, epsilons, color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "alpha_over_time.png")
    )
    plot_epsilon_over_time(
        [(name, alphas, epsilons, color)
         for (name, succ, rewards, refl, alphas, epsilons), color in zip(results, ["blue","purple","red","green"])],
        save=os.path.join(save_dir, "epsilon_over_time.png")
    )

    # Write summary
    write_summary_md(
        results,
        os.path.join(save_dir, "summary.md")
    )