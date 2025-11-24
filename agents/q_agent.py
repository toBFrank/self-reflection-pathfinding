import numpy as np
from collections import deque

import os, math, random, argparse, time
from collections import defaultdict, deque
from datetime import datetime
import matplotlib.pyplot as plt

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
    
class QLearner:
    def __init__(self, actions=4, alpha=0.6, gamma=0.95, epsilon=0.1, epsilon_min=0.05, epsilon_decay=0.999, dyna_k=10, seed=None):
        self.actions = actions       # 4 for baseline, 5 for adaptive (with REFLECT action)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.dyna_k = dyna_k
        self.Q = defaultdict(lambda: np.zeros(self.actions))
        self.buffer = []             # replay (s,a,r,s2,done); used by dyna_reflect
        self.rng = random.Random(seed)
        self.returns = deque(maxlen=20)

    def choose_action(self, s):
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        return int(np.argmax(self.Q[s]))

    def q_update(self, s, a, r, s2, done):
        q_next = 0.0 if done else np.max(self.Q[s2])
        self.Q[s][a] += self.alpha * (r + self.gamma * q_next - self.Q[s][a])

    def remember(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
        if len(self.buffer) > 50000:
            self.buffer = self.buffer[-25000:]

    def dyna_reflect(self):
        # offline planning updates
        for _ in range(self.dyna_k):
            if not self.buffer: break
            s, a, r, s2, done = self.rng.choice(self.buffer)
            self.q_update(s, a, r, s2, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


class AgentWrapper:
    """
    A unified interface for:
      - Baseline Q-learning (no reflection)
      - Reflective Q-learning (5 actions)
      - Dyna-Reflective Q-learning (QLearner with planning)
    """

    def __init__(self, env, mode="baseline", reflection_strategy=None):
        """
        mode:
            "baseline"   -> standard Q-learning, no reflect action
            "reflective" -> Q-learning, uses reflect action
            "dyna"       -> QLearner with Dyna + reflect action

        reflection_strategy: object with .reflect(agent) method
        """
        self.mode = mode
        self.env = env
        self.reflection_strategy = reflection_strategy

        # select agent type
        if mode == "baseline":
            self.agent = QAgent(env,
                                use_reflection=False,
                                reflection_strategy=None)

        elif mode == "reflective":
            self.agent = QAgent(env,
                                use_reflection=True,
                                reflection_strategy=reflection_strategy)

        elif mode == "dyna":
            # QLearner uses dict Q-table and supports Dyna
            self.agent = QLearner(
                actions=5,       # movement + REFLECT
                alpha=0.6,
                gamma=0.95,
                epsilon=0.1,
                epsilon_min=0.05,
                epsilon_decay=0.999,
                dyna_k=10
            )

        else:
            raise ValueError(f"Unknown mode '{mode}'")

    # Public training interface (compatible with run_experiment)
    def train_episode(self):
        if self.mode in ("baseline", "reflective"):
            return self.agent.train_episode()

        elif self.mode == "dyna":
            return self._train_episode_dyna()

    # Dyna Reflection Episode Loop
    def _train_episode_dyna(self):
        total_reward = 0
        reflections_used = 0

        # QLearner uses tuple states
        s = tuple(self.env.reset())
        done = False

        while not done:
            # action
            a = self.agent.choose_action(s)

            # take step
            s2, reward, done = self.env.step(a)
            s2 = tuple(s2)
            total_reward += reward

            # store transition in replay buffer
            self.agent.remember(s, a, reward, s2, done)

            # standard Q-learning update
            self.agent.q_update(s, a, reward, s2, done)

            # reflection logic
            if a == self.env.ACTION_REFLECT:
                reflections_used += 1
                if self.reflection_strategy:
                    self.reflection_strategy.reflect(self.agent)

            # dyna offline updates
            self.agent.dyna_reflect()

            # decay exploration rate
            self.agent.decay_epsilon()

            # move to next state
            s = s2
            self.agent.returns.append(total_reward)


        return total_reward, reflections_used
