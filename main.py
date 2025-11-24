import matplotlib.pyplot as plt
from agents.reflection.epsilon_reflection import EpsilonReflection
from experiment.gridworld import GridWorld
from agents.q_agent import QAgent, AgentWrapper
from experiment.runner import run_experiment

# Create environment
env = GridWorld(size=11, reflect_cost=0.05)
env.generate_environment(num_obstacles=3)

# Experiment 0: Q-learning, no reflection (Baseline)
print("Q-Learning, No Reflection (Baseline)")
#baseline_agent = QAgent(env, use_reflection=False)
baseline_agent = AgentWrapper(env, mode="baseline")
baseline_rewards = run_experiment(env, baseline_agent, episodes=500)

# Experiment 1: Q-learning, epsilon reflection, low cost
print("\nQ-Learning, Epsilon Reflection, Low Cost ($)")
reflect_agent = AgentWrapper(env, mode="reflective", reflection_strategy=EpsilonReflection())
reflect_rewards = run_experiment(env, reflect_agent, episodes=500)

# Experiment 1.5: Dyna-Q, epsilon reflection, low cost
print("\nDyna-Q, Epsilon Reflection, Low Cost ($)")
dyna_agent = AgentWrapper(env, mode="dyna", reflection_strategy=EpsilonReflection())
dyna_rewards = run_experiment(env, dyna_agent, episodes=500)

# Experiment 2: Q-learning, epsilon reflection, medium cost
print("\nQ-Learning, Epsilon Reflection, Medium Cost ($$)")
print("Not implemented in this run.")

# Experiment 3: Q-learning, epsilon reflection, high cost
print("\nQ-Learning, Epsilon Reflection, High Cost ($$$)")
print("Not implemented in this run.")

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(baseline_rewards, label="No Reflection")
plt.plot(reflect_rewards, label="Reflection Enabled")
plt.plot(dyna_rewards, label="Dyna + Reflection")
plt.title("Learning Curves")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()
