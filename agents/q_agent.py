import numpy as np
from collections import deque

class QAgent:
    def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.2, use_reflection=True, reflection_strategy=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_reflection = use_reflection
        self.reflection_strategy = reflection_strategy

        self.n_actions = 5 if use_reflection else 4
        self.Q = np.zeros((env.size, env.size, self.n_actions))

        self.returns = deque(maxlen=20)

    def choose_action(self, state):
        x,y = state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[x,y])

    def train_episode(self):
        self.num_reflections_used = 0
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            x,y = state
            action = self.choose_action(state)

            next_state, reward, done = self.env.step(action)
            nx,ny = next_state
            total_reward += reward

            best_next = np.max(self.Q[nx,ny])
            self.Q[x,y,action] += self.alpha * (
                reward + self.gamma * best_next - self.Q[x,y,action]
            )

            if self.use_reflection and action == self.env.ACTION_REFLECT:
                if self.reflection_strategy:
                    self.reflection_strategy.reflect(self)
                self.num_reflections_used += 1

            state = next_state

        self.returns.append(total_reward)
        return total_reward, self.num_reflections_used
