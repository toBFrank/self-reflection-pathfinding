"""
Self-Reflection Project (Professor-aligned)
Baseline = pure Q-learning (4 actions)
Adaptive = Q-learning + REFLECT meta-action (5th action):
 - choosing REFLECT: reward = -cost, state doesn't change, perform K Dyna planning updates from replay

Env:
 - 20x20 grid, 10% obstacles, random start/goal with Manhattan >= 1.4*side
 - rewards: +10 goal, -1 if no movement (bump/blocked), -0.1 per step
 - map resampled every 5 episodes

Outputs:
 - outputs/<timestamp>/cost_<c>/{learning_curves.png, reflect_freq.png, logs.csv}
"""

import os, math, random, argparse, time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ---- Constants
UP, RIGHT, DOWN, LEFT, REFLECT = 0, 1, 2, 3, 4

# =========================
# Environment
# =========================
class GridWorld:
    def __init__(
        self,
        size=20,
        obstacle_density=0.10,
        resample_every=5,
        step_penalty=-0.1,
        bump_penalty=-1.0,
        goal_reward=10.0,
        max_steps=800,
        seed=None,
    ):
        self.size = size
        self.obstacle_density = obstacle_density
        self.resample_every = resample_every
        self.step_penalty = step_penalty
        self.bump_penalty = bump_penalty
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.episode_count = 0

        self.randomize_map()  # sets start, goal, obstacles
        self.reset()

    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _sample_start_goal(self):
        thr = math.ceil(1.4 * self.size)
        cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        start = self.rng.choice(cells)
        goal = self.rng.choice(cells)
        while self.manhattan(start, goal) < thr:
            start = self.rng.choice(cells)
            goal = self.rng.choice(cells)
        return start, goal

    def _place_obstacles(self, start, goal):
        total = self.size * self.size
        k = int(round(self.obstacle_density * total))
        all_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        candidates = [c for c in all_cells if c != start and c != goal]
        self.rng.shuffle(candidates)
        obstacles = set(candidates[:k])
        return obstacles

    def _path_exists(self, start, goal, obstacles):
        # BFS to ensure reachability
        q = deque([start])
        seen = {start}
        dirs = [(-1,0),(0,1),(1,0),(0,-1)]
        while q:
            x,y = q.popleft()
            if (x,y) == goal: return True
            for dx,dy in dirs:
                nx,ny = x+dx, y+dy
                if 0<=nx<self.size and 0<=ny<self.size and (nx,ny) not in obstacles:
                    if (nx,ny) not in seen:
                        seen.add((nx,ny))
                        q.append((nx,ny))
        return False

    def randomize_map(self):
        # resample until start-goal far enough and path exists
        tries = 0
        while True:
            tries += 1
            start, goal = self._sample_start_goal()
            obstacles = self._place_obstacles(start, goal)
            if self._path_exists(start, goal, obstacles):
                self.start, self.goal, self.obstacles = start, goal, obstacles
                break
            if tries > 100:  # very unlikely
                # fallback: reduce obstacles
                self.obstacle_density = max(0.05, self.obstacle_density * 0.8)
        self.resample_anchor = self.episode_count

    def reset(self):
        # resample map every resample_every episodes
        if self.episode_count % self.resample_every == 0 and self.episode_count != 0:
            self.randomize_map()
        self.pos = self.start
        self.steps = 0
        self.episode_count += 1
        return self.pos

    def step(self, action):
        x, y = self.pos
        if action == UP:      nx, ny = x-1, y
        elif action == RIGHT: nx, ny = x, y+1
        elif action == DOWN:  nx, ny = x+1, y
        elif action == LEFT:  nx, ny = x, y-1
        else:                 nx, ny = x, y  # invalid=stay

        # attempt move
        moved = False
        if 0 <= nx < self.size and 0 <= ny < self.size and (nx,ny) not in self.obstacles:
            if (nx,ny) != self.pos:
                moved = True
            self.pos = (nx, ny)

        self.steps += 1

        # reward logic
        if self.pos == self.goal:
            return self.pos, self.goal_reward, True

        if not moved:  # bump / no movement
            reward = self.bump_penalty
        else:
            reward = self.step_penalty

        done = self.steps >= self.max_steps
        return self.pos, reward, done


# =========================
# Agent (Q-learning + optional REFLECT)
# =========================
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


# =========================
# Training
# =========================
def run_episode(env, agent, allow_reflect=False, reflect_cost=0.0):
    s = env.reset()
    total_reward = 0.0
    reflect_steps = 0
    done = False
    steps = 0                    # <-- new: count all steps, including REFLECT

    while not done and steps < env.max_steps:
        a = agent.choose_action(s)

        if allow_reflect and agent.actions == 5 and a == REFLECT:
            # REFLECT: stay in place, pay cost, do planning
            agent.q_update(s, REFLECT, -reflect_cost, s, False)
            agent.dyna_reflect()
            total_reward += -reflect_cost
            reflect_steps += 1
            s2 = s
            done = False
        else:
            # normal environment step
            s2, r, done = env.step(a)
            agent.q_update(s, a, r, s2, done)
            agent.remember(s, a, r, s2, done)
            total_reward += r

        s = s2
        steps += 1               # <-- count both REFLECT and env steps

    agent.decay_epsilon()
    return total_reward, reflect_steps



def train_compare(args, outdir, seed):
    # single-seed run for all costs; returns dict of logs per cost
    rng = np.random.default_rng(seed)
    base_env = GridWorld(size=20, obstacle_density=0.10, resample_every=5, seed=seed)
    # we re-create agents/envs per cost (fair comparison)
    cost_logs = {}

    for cost in args.costs:
        # Re-seed reproducibly each cost
        env = GridWorld(size=20, obstacle_density=0.10, resample_every=5, seed=seed)
        baseline = QLearner(actions=4, alpha=args.alpha, gamma=args.gamma,
                            epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay, dyna_k=0, seed=seed)

        adaptive = QLearner(actions=5, alpha=args.alpha, gamma=args.gamma,
                            epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                            epsilon_decay=args.epsilon_decay, dyna_k=args.dyna_k, seed=seed)

        base_returns, adap_returns, refl_counts = [], [], []

        for ep in range(args.episodes):
            Rb, _ = run_episode(env, baseline, allow_reflect=False, reflect_cost=0.0)
            Ra, rc = run_episode(env, adaptive, allow_reflect=True, reflect_cost=cost)
            base_returns.append(Rb)
            adap_returns.append(Ra)
            refl_counts.append(rc)

        cost_logs[cost] = {
            "baseline_returns": np.array(base_returns, dtype=float),
            "adaptive_returns": np.array(adap_returns, dtype=float),
            "reflect_counts":   np.array(refl_counts, dtype=int),
        }

    return cost_logs


def aggregate_and_save(all_logs, args, outdir):
    os.makedirs(outdir, exist_ok=True)

    # For each cost, aggregate across seeds and save CSV + plots
    for cost, seed_logs in all_logs.items():
        # Stack seeds
        base_mat = np.stack([seed_logs[s]["baseline_returns"] for s in seed_logs])
        adap_mat = np.stack([seed_logs[s]["adaptive_returns"] for s in seed_logs])
        refl_mat = np.stack([seed_logs[s]["reflect_counts"]   for s in seed_logs])

        def mean_ci(mat):
            mean = mat.mean(axis=0)
            if mat.shape[0] > 1:
                # 95% CI via normal approx
                std = mat.std(axis=0, ddof=1)
                ci = 1.96 * std / np.sqrt(mat.shape[0])
            else:
                ci = np.zeros_like(mean)
            return mean, ci

        base_mean, base_ci = mean_ci(base_mat)
        adap_mean, adap_ci = mean_ci(adap_mat)
        refl_mean, refl_ci = mean_ci(refl_mat)

        cost_dir = os.path.join(outdir, f"cost_{cost}")
        os.makedirs(cost_dir, exist_ok=True)

        # Save CSV
        csv_path = os.path.join(cost_dir, "logs.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("episode,baseline_return,adaptive_return,reflect_count,baseline_ci,adaptive_ci,reflect_ci\n")
            for i in range(base_mean.shape[0]):
                f.write(f"{i},{base_mean[i]:.6f},{adap_mean[i]:.6f},{refl_mean[i]:.6f},{base_ci[i]:.6f},{adap_ci[i]:.6f},{refl_ci[i]:.6f}\n")

        # Plots
        # Learning curves
        plt.figure(figsize=(8,5))
        plt.plot(base_mean, label="Baseline (no reflection)")
        plt.fill_between(range(len(base_mean)), base_mean-base_ci, base_mean+base_ci, alpha=0.2)
        plt.plot(adap_mean, label=f"Adaptive (reflect cost={cost})")
        plt.fill_between(range(len(adap_mean)), adap_mean-adap_ci, adap_mean+adap_ci, alpha=0.2)
        plt.xlabel("Episode")
        plt.ylabel("Net Episodic Return")
        plt.title(f"Learning Curves (cost={cost})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cost_dir, "learning_curves.png"))
        plt.close()

        # Reflection frequency
        plt.figure(figsize=(8,4))
        plt.plot(refl_mean, label="Reflect steps / episode")
        plt.fill_between(range(len(refl_mean)), refl_mean-refl_ci, refl_mean+refl_ci, alpha=0.2)
        plt.xlabel("Episode")
        plt.ylabel("Reflection count")
        plt.title(f"Reflection Frequency (cost={cost})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cost_dir, "reflect_freq.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="episodes per run")
    parser.add_argument("--seeds", type=int, default=5, help="number of random seeds")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--epsilon-min", type=float, default=0.05, dest="epsilon_min")
    parser.add_argument("--epsilon-decay", type=float, default=0.999, dest="epsilon_decay")
    parser.add_argument("--dyna-k", type=int, default=10, help="planning updates per REFLECT")
    parser.add_argument("--costs", type=float, nargs="+", default=[0.02, 0.20, 1.0], help="reflect cost levels")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("outputs", stamp)
    os.makedirs(outdir, exist_ok=True)

    # Run all seeds
    all_logs = {c:{} for c in args.costs}
    for s in range(args.seeds):
        print(f"[Seed {s+1}/{args.seeds}] running...")
        cost_logs = train_compare(args, outdir, seed=s)
        for c in args.costs:
            all_logs[c][s] = cost_logs[c]

    # Aggregate across seeds and save
    aggregate_and_save(all_logs, args, outdir)

    print(f"\nDone. Results saved under: {outdir}")
    for c in args.costs:
        print(f"  - cost_{c}/learning_curves.png")
        print(f"  - cost_{c}/reflect_freq.png")
        print(f"  - cost_{c}/logs.csv")

if __name__ == "__main__":
    main()
