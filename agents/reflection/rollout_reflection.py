import numpy as np
from .reflection_strategy import ReflectionStrategy

class RolloutReflection(ReflectionStrategy):
    """
    Runs several imaginary rollouts and updates Q as if learning from them.
    """

    def __init__(self, depth=5, n_rollouts=10, discount=0.95):
        self.depth = depth
        self.n_rollouts = n_rollouts
        self.discount = discount

    def reflect(self, agent):
        env = agent.env

        for _ in range(self.n_rollouts):
            self.run_rollout(agent, env)

    def run_rollout(self, agent, env):
        simulated_environment = env.copy()

        x, y = env.agent_pos

        total_return = 0
        discount = 1.0
        action = None

        for t in range(self.depth):
            a = np.random.randint(4)  # movement only (UP, RIGHT, DOWN, LEFT)
            if t == 0:
                # first action from current state
                action = a

            next_state, reward, done = simulated_environment.step(a)
            total_return += discount * reward
            discount *= self.discount

            if done:
                break

        # update Q for current state
        agent.Q[x,y,action] += agent.alpha * (total_return - agent.Q[x,y,action])