import os
from experiment.gridworld import GridWorld
from agents.q_agent import QAgent
from agents.reflection.adaptive_reflection import AdaptiveReflection
from agents.reflection.reflection_costs import ReflectionCost
from experiment.runner import run_experiment_with_paths
from analysis.plotting import plot_reflection_comparison_videos

if __name__ == "__main__":
    env = GridWorld(size=20, min_distance=15)
    env.generate_environment(num_obstacles=100)

    runs = []
    for name, cost in [("baseline", None), ("low", ReflectionCost.LOW.value), ("medium", ReflectionCost.MEDIUM.value), ("high", ReflectionCost.HIGH.value)]:
        if cost is not None:
            env.set_reflect_cost(cost)
        else:
            env.set_reflect_cost(0.0)
        agent = QAgent(env, use_reflection=(cost is not None), reflection_strategy=AdaptiveReflection())
        _, _, _, _, _, _, _, successful_paths = run_experiment_with_paths(env, agent, episodes=80)
        runs.append((name, successful_paths))

    out_dir = os.path.join("results", "video_demo")
    plot_reflection_comparison_videos(env, runs, save_dir=out_dir, fps=6, interval=500, pause_frames=8)
    print(f"Saved comparison videos to {out_dir}")
