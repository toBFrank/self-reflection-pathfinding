import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class GridWorld:
    # Action definitions
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_REFLECT = 4  # reflection action

    def __init__(self, size=5, reflect_cost=0.0, min_distance=4):
        self.size = size
        self.reflect_cost = reflect_cost
        self.min_distance = min_distance

        # grid: 0 empty, 1 obstacle
        self.grid = np.zeros((size, size))

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []

        self.num_obstacles = 20

    # environment generation methods

    def manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def is_path_exists(self, start, end):
        """BFS to check if path exists from start to end avoiding obstacles"""
        from collections import deque

        if start == end:
            return True
        if self.grid[start] == 1 or self.grid[end] == 1:
            return False

        queue = deque([start])
        visited = {start}

        dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        while queue:
            x,y = queue.popleft()
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx,ny] == 0 and (nx,ny) not in visited:
                        if (nx,ny) == end:
                            return True
                        visited.add((nx,ny))
                        queue.append((nx,ny))

        return False

    def set_agent_goal(self):
        positions = [(i,j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(positions)

        for a in positions:
            valid_goals = [g for g in positions if self.manhattan_distance(a,g) >= self.min_distance]
            if not valid_goals:
                continue
            random.shuffle(valid_goals)
            for g in valid_goals:
                if self.is_path_exists(a,g):
                    self.agent_pos = a
                    self.goal_pos = g
                    return True
        return False

    def generate_obstacles(self, num_obstacles):
        self.obstacles = []
        free_positions = [(i,j) for i in range(self.size) for j in range(self.size)
                          if (i,j) != self.agent_pos and (i,j) != self.goal_pos]
        random.shuffle(free_positions)

        for pos in free_positions[:num_obstacles]:
            self.grid[pos] = 1
            self.obstacles.append(pos)

    def generate_environment(self, num_obstacles=None):
        if num_obstacles is None:
            num_obstacles = self.num_obstacles  # 使用实例变量作为默认值
            self.grid = np.zeros((self.size, self.size))

        if not self.set_agent_goal():
            raise Exception("could not place agent and goal")

        # place obstacles
        self.generate_obstacles(num_obstacles)

        # ensure solvable
        if not self.is_path_exists(self.agent_pos, self.goal_pos):
            # try fewer obstacles
            self.grid = np.zeros((self.size, self.size))
            self.obstacles = []
            self.generate_obstacles(max(0, num_obstacles - 1))

    # environment interaction methods

    def reset(self):
        """start a new episode. agent stays at original spawn."""
        self.generate_environment()
        return self.agent_pos

    def step(self, action):
        """
        takes an action and returns (next_state, reward, done)
        """

        # reflection action
        if action == self.ACTION_REFLECT:
            reward = -0.1 - self.reflect_cost
            return self.agent_pos, reward, False

        # movement actions
        x,y = self.agent_pos
        if action == self.ACTION_UP:
            nx,ny = x-1, y
        elif action == self.ACTION_RIGHT:
            nx,ny = x, y+1
        elif action == self.ACTION_DOWN:
            nx,ny = x+1, y
        elif action == self.ACTION_LEFT:
            nx,ny = x, y-1
        else:
            raise ValueError("Invalid action.")

        # check legality
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            return self.agent_pos, -1.0, False
        if self.grid[nx,ny] == 1:
            return self.agent_pos, -1.0, False

        # valid move
        self.agent_pos = (nx, ny)

        # check goal
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10.0, True

        # movement penalty
        return self.agent_pos, -0.1, False



# QAgent with Reflection Mechanism

class QAgent:
    def __init__(self, env, alpha=0.3, gamma=0.95, epsilon=0.2, use_reflection=True):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_reflection = use_reflection

        self.n_actions = 5 if use_reflection else 4
        self.Q = np.zeros((env.size, env.size, self.n_actions))

        self.returns = deque(maxlen=20)

    def choose_action(self, state):
        x,y = state
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[x,y])

    def reflect(self):
        """Updates epsilon based on performance trend."""
        if len(self.returns) < 10:
            return

        early = np.mean(list(self.returns)[:5])
        late = np.mean(list(self.returns)[5:])
        if late < early:
            self.epsilon = min(1.0, self.epsilon + 0.02)
        else:
            self.epsilon = max(0.01, self.epsilon - 0.02)

    def train_episode(self, max_steps=50000):  # 添加 max_steps 参数
        self.num_reflections_used = 0
        self.num_moves = 0
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0  # 新增计数器

        while not done and steps < max_steps:  # ← 加入步数限制
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            total_reward += reward

            nx, ny = next_state
            x, y = state

            best_next = np.max(self.Q[nx, ny])
            self.Q[x, y, action] += self.alpha * (reward + self.gamma * best_next - self.Q[x, y, action])

            if self.use_reflection and action == self.env.ACTION_REFLECT:
                self.reflect()
                self.num_reflections_used += 1
            else:
                self.num_moves += 1

            state = next_state
            steps += 1  # 每次循环都算一步（包括 reflection）


        self.returns.append(total_reward)
        return total_reward, self.num_reflections_used, self.num_moves


# experiment runner (modified to record reflection usage per episode)
def run_experiment(env, agent, episodes=50000, max_steps=500):
    rewards = []
    reflect_counts = []
    move_steps = []
    for ep in range(episodes):
        r, refl, moves = agent.train_episode(max_steps=max_steps)
        rewards.append(r)
        reflect_counts.append(refl)
        move_steps.append(moves)
        if refl > 0:
            print(f"Episode {ep} | Reward = {r:.2f} | Reflections = {refl} | Steps = {moves}")
    return rewards, reflect_counts, move_steps

# main experiment

# create environment
env = GridWorld(size=11, reflect_cost=0.05)
env.generate_environment(num_obstacles=20)

# baseline (NO reflection)
print("NO REFLECTION")
baseline_agent = QAgent(env, use_reflection=False)
baseline_rewards, _, baseline_moves = run_experiment(env, baseline_agent, episodes=500, max_steps=50000)

# reflective agent
print("\nWITH REFLECTION")
reflect_agent = QAgent(env, use_reflection=True)
reflect_rewards, reflect_counts, reflect_moves = run_experiment(env, reflect_agent, episodes=500, max_steps=50000)

import matplotlib.pyplot as plt

reflect_count_mean = np.mean(reflect_counts)
baseline_reward_mean = np.mean(baseline_rewards)
reflect_reward_mean = np.mean(reflect_rewards)
baseline_moves_mean = np.mean(baseline_moves)
reflect_moves_mean = np.mean(reflect_moves)

# 创建画布：3 个子图
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 图1：反思次数柱状图（保持不变）
axes[0].bar(range(len(reflect_counts)), reflect_counts, color='skyblue', alpha=0.7)
axes[0].set_ylabel('Reflection Count')
axes[0].set_title('Reflection Usage per Episode')
axes[0].grid(alpha=0.3)

# 添加总反思次数的平均值
axes[0].axhline(reflect_count_mean, color='r', linestyle='--', linewidth=2,
                label=f'Avg Reflection Count: {reflect_count_mean:.2f}')
axes[0].legend()

# 图2：学习曲线（总奖励）
axes[1].plot(baseline_rewards, label="No Reflection", alpha=0.8)
axes[1].plot(reflect_rewards, label="With Reflection", alpha=0.8)

# 添加总奖励的平均线
axes[1].axhline(baseline_reward_mean, color='C0', linestyle='--', linewidth=2,
                label=f'Avg No Reflection: {baseline_reward_mean:.2f}')
axes[1].axhline(reflect_reward_mean, color='C1', linestyle='--', linewidth=2,
                label=f'Avg With Reflection: {reflect_reward_mean:.2f}')

axes[1].set_ylabel('Total Reward')
axes[1].set_title('Learning Curves (Total Reward)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# 图3：路径长度对比
axes[2].plot(baseline_moves, label="No Reflection", alpha=0.8)
axes[2].plot(reflect_moves, label="With Reflection", alpha=0.8)

# 添加路径长度的平均线
axes[2].axhline(baseline_moves_mean, color='C0', linestyle='--', linewidth=2,
                label=f'Avg No Reflection: {baseline_moves_mean:.1f}')
axes[2].axhline(reflect_moves_mean, color='C1', linestyle='--', linewidth=2,
                label=f'Avg With Reflection: {reflect_moves_mean:.1f}')

axes[2].set_xlabel('Episode')
axes[2].set_ylabel('Path Length (Steps)')
axes[2].set_title('Path Length Comparison per Episode')
axes[2].legend()
axes[2].grid(alpha=0.3)

# 自动调整布局，避免重叠
plt.tight_layout()
plt.show()
