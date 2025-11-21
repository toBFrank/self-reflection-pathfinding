import matplotlib.pyplot as plt
from env.gridworld import GridWorld
from agents.q_agent import QAgent
from experiment.runner import run_experiment

# Create environment
env = GridWorld(size=11, reflect_cost=0.05)
env.generate_environment(num_obstacles=3)

# Baseline (no reflection)
print("NO REFLECTION")
baseline_agent = QAgent(env, use_reflection=False)
baseline_rewards = run_experiment(env, baseline_agent, episodes=500)

# Reflective agent
print("\nWITH REFLECTION")
reflect_agent = QAgent(env, use_reflection=True)
reflect_rewards = run_experiment(env, reflect_agent, episodes=500)

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(baseline_rewards, label="No Reflection")
plt.plot(reflect_rewards, label="Reflection Enabled")
plt.title("Learning Curves")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()
