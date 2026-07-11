import numpy as np
from collections import deque

class QAgent:
    def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.2, use_reflection=True, reflection_strategy=None):
        """
        Q-Learning agent that can self-reflect.

        Reflection is a special action that agent can optionally choose to take.
        When reflecting, it uses a reflection strategy that changes the agent's internal parameters (e.g., epsilon).

        Args:
            env: The environment the agent is put in (gridworld).
            alpha: Q-learning rate (high = trust new information more, low = trust old information more).
            gamma: Discount factor (high = future rewards more valuable, low = immediate rewards more valuable).
            epsilon: Exploration rate (high = more random moves, low = less random moves).
            use_reflection: Whether to enable reflection action.
            reflection_strategy: An instance of a ReflectionStrategy to use when reflecting.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_reflection = use_reflection
        self.reflection_strategy = reflection_strategy

        self.alphas = [self.alpha]
        self.epsilons = [self.epsilon]

        self.n_actions = 5 if use_reflection else 4
        self.Q = np.zeros((env.size, env.size, self.n_actions))

        self.returns = deque(maxlen=20)

    def choose_action(self, state, invalid_actions=[]):
        x,y = state
        valid_actions = []
        if np.random.rand() < self.epsilon:
            for a in range(self.n_actions):
                if a not in invalid_actions:
                    valid_actions.append(a)
            return np.random.choice(valid_actions)
        # if np.argmax(self.Q[x,y]) in invalid_actions, choose the next best action
        sorted_actions = np.argsort(self.Q[x,y])[::-1]  # descending order
        for action in sorted_actions:
            if action not in invalid_actions:
                return action
        return np.argmax(self.Q[x,y])

    def train_episode(self, max_steps=10000):
        self.num_reflections_used = 0
        state = self.env.reset()
        done = False
        total_reward = 0
        path = [state]

        for step in range(max_steps):
            x,y = state
            invalid_actions = []
            while True:
                action = self.choose_action(state, invalid_actions)
                # if the agent hits a wall or obstacle, reward == 0 and it must choose again
                next_state, reward, done = self.env.step(action)
                if reward != 0 or action == self.env.ACTION_REFLECT:
                    break
                else:
                    invalid_actions.append(action)
        
            invalid_actions = []
            path.append(next_state)
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

            if done:
                # print("Reached goal!")
                break

            state = next_state

        if not done:
            # print("Goal not reached within step limit.")
            pass
        self.returns.append(total_reward)
        self.alphas.append(self.alpha)
        self.epsilons.append(self.epsilon)
        return total_reward, self.num_reflections_used, path, done, self.alphas, self.epsilons