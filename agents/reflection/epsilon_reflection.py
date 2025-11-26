import numpy as np
from .reflection_strategy import ReflectionStrategy

class EpsilonReflection(ReflectionStrategy):
    """
    Updates the agent's epsilon based on recent performance.
    If performance gets better, decrease epsilon (less exploration).
    If performance gets worse, increase epsilon (more exploration).
    """
    def __init__(self, delta=0.02, window=10):
        self.delta = delta
        self.window = window

    def reflect(self, agent):
        if len(agent.returns) < self.window:
            return

        early = np.mean(list(agent.returns)[:5])
        late  = np.mean(list(agent.returns)[5:])

        # degrade or improve exploration
        if late < early:
            agent.epsilon = min(1.0, agent.epsilon + self.delta)
        else:
            agent.epsilon = max(0.01, agent.epsilon - self.delta)
