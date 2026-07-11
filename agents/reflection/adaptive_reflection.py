from collections import deque

import numpy as np
from .reflection_strategy import ReflectionStrategy

class AdaptiveReflection(ReflectionStrategy):
    def __init__(self, window_size=10):
        # original gamma: 0.95, epsilon: 0.2, alpha: 0.3
        self.returns = deque(maxlen=window_size)
        self.prev_avg_return = None
        self.window_size = window_size

    def reflect(self, agent):
        # Get average return of recent episodes (including this one)
        if len(agent.returns) < self.window_size:
            return  # no data yet
        
        # previous avg = mean of previous 5 returns
        self.prev_avg_return = np.mean(list(agent.returns)[-10:-5])
        # current avg = mean of last 5 returns
        current_avg = np.mean(list(agent.returns)[-5:])

        # if performance improved, decrease epsilon (less exploration)
        # if performance improved, increase alpha (trust new info more)
        if current_avg >= self.prev_avg_return:
            agent.epsilon = max(agent.epsilon - 0.02, 0.01)
            agent.alpha = min(agent.alpha + 0.02, 1)
        else:
            agent.epsilon = min(agent.epsilon + 0.01, 1.0)
            agent.alpha = max(agent.alpha - 0.02, 0.05)