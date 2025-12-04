import numpy as np
import matplotlib.pyplot as plt
import random

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

        self.start_pos = None
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []

    def set_reflect_cost(self, cost):
        self.reflect_cost = cost

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
                    self.start_pos = a
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

    def generate_environment(self, num_obstacles=30, max_attempts=1000):
        self.grid = np.zeros((self.size, self.size))

        # set start + goal (grid is empty so guaranteed solvable)
        if not self.set_agent_goal():
            raise Exception("could not place agent and goal")

        attempt = 0
        while attempt < max_attempts:
            attempt += 1

            # Start with empty grid every attempt
            self.grid[:, :] = 0
            self.obstacles = []

            # generate obstacles
            self.generate_obstacles(num_obstacles)

            # check solvability
            if self.is_path_exists(self.start_pos, self.goal_pos):
                return  # success!

        # if we reach here → we failed after max_attempts
        raise Exception("Could not create a solvable map after repeated attempts.")

    # def generate_environment(self, num_obstacles=30):
    #     self.grid = np.zeros((self.size, self.size))

    #     if not self.set_agent_goal():
    #         raise Exception("could not place agent and goal")

    #     # place obstacles
    #     self.generate_obstacles(num_obstacles)

    #     # ensure solvable
    #     if not self.is_path_exists(self.agent_pos, self.goal_pos):
    #         # try fewer obstacles
    #         self.grid = np.zeros((self.size, self.size))
    #         self.obstacles = []
    #         self.generate_obstacles(max(0, num_obstacles - 1))

    # environment interaction methods
    def reset(self):
        """start a new episode. agent stays at original spawn."""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        """
        takes an action and returns (next_state, reward, done)
        """

        old_pos = self.agent_pos

        # reflection action
        if action == self.ACTION_REFLECT:
            # Reflection cost only
            return old_pos, 0.05 - self.reflect_cost, False

        x,y = old_pos
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

        # wall or obstacle
        if not (0 <= nx < self.size and 0 <= ny < self.size) or self.grid[nx,ny] == 1:
            return old_pos, 0, False

        # update position
        self.agent_pos = (nx, ny)
        new_dist = self.manhattan_distance(self.agent_pos, self.goal_pos)

        # goal reward
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 10.0, True

        # step cost
        reward = -0.1

        return self.agent_pos, reward, False

    # utility methods
    def copy(self):
        new_env = GridWorld(size=self.size, reflect_cost=self.reflect_cost, min_distance=self.min_distance)
        new_env.grid = np.copy(self.grid)
        new_env.start_pos = self.start_pos
        new_env.agent_pos = self.agent_pos
        new_env.goal_pos = self.goal_pos
        new_env.obstacles = list(self.obstacles)
        return new_env

    def visualize_grid(self):
        grid = np.zeros((self.size, self.size, 3))  # RGB grid

        # Empty = white
        grid[:] = [1, 1, 1]

        # Obstacles = red
        for ox, oy in self.obstacles:
            grid[ox, oy] = [1, 0, 0]

        # Goal = green
        gx, gy = self.goal_pos
        grid[gx, gy] = [0, 1, 0]

        # Agent = blue
        ax, ay = self.agent_pos
        grid[ax, ay] = [0, 0, 1]

        plt.imshow(grid)
        plt.title("GridWorld")
        plt.grid(True, color="black", linewidth=0.5)
        plt.xticks(np.arange(-.5, self.size, 1))
        plt.yticks(np.arange(-.5, self.size, 1))
        plt.show()

    def plot_path(self, path):
        """
        - Start position = blue
        - Goal = green
        - Obstacles = red
        - Current (final) agent position = yellow
        - Path = black line
        """

        grid = np.zeros((self.size, self.size, 3))
        grid[:] = [1, 1, 1]  # white background

        # Obstacles = red
        for ox, oy in self.obstacles:
            grid[ox, oy] = [1, 0, 0]

        # Goal = green
        gx, gy = self.goal_pos
        grid[gx, gy] = [0, 1, 0]

        # Start position = blue
        sx, sy = path[0]
        grid[sx, sy] = [0, 0, 1]

        # Final position = yellow
        fx, fy = path[-1]
        grid[fx, fy] = [1, 1, 0]   # yellow

        # --- Plotting ---
        plt.figure(figsize=(6, 6))
        plt.imshow(grid)
        plt.title("Agent Path")

        # Path line
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys, marker="o", markersize=4, linewidth=1, color="black")

        # Draw grid lines
        plt.grid(True, color="black", linewidth=0.5)
        plt.xticks(np.arange(-.5, self.size, 1))
        plt.yticks(np.arange(-.5, self.size, 1))

        plt.show()

