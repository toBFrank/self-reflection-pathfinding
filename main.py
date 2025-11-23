import matplotlib.pyplot as plt
from agents.reflection.epsilon_reflection import EpsilonReflection
from experiment.gridworld import GridWorld
from agents.q_agent import QAgent
from experiment.runner import run_experiment

# Create environment
env = GridWorld(size=11, reflect_cost=0.05)
env.generate_environment(num_obstacles=3)

# Experiment 0: Q-learning, no reflection (Baseline)
print("Q-Learning, No Reflection (Baseline)")
baseline_agent = QAgent(env, use_reflection=False)
baseline_rewards = run_experiment(env, baseline_agent, episodes=500)

# Experiment 1: Q-learning, epsilon reflection, low cost
print("\nQ-Learning, Epsilon Reflection, Low Cost ($)")
reflect_agent = QAgent(env, use_reflection=True, reflection_strategy=EpsilonReflection())
reflect_rewards = run_experiment(env, reflect_agent, episodes=500)

# Experiment 2: Q-learning, epsilon reflection, medium cost
print("\nQ-Learning, Epsilon Reflection, Medium Cost ($$)")
print("Not implemented in this run.")

# Experiment 3: Q-learning, epsilon reflection, high cost
print("\nQ-Learning, Epsilon Reflection, High Cost ($$$)")
print("Not implemented in this run.")

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.scatter(range(len(baseline_rewards)), baseline_rewards, label="Agent No Reflection", s=10)  # s= marker size
plt.scatter(range(len(reflect_rewards)), reflect_rewards, label="Agent Reflection Enabled", s=10)
# plt.plot(baseline_rewards, label="No Reflection")
# plt.plot(reflect_rewards, label="Reflection Enabled")

plt.title("Reward Trends")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.show()
