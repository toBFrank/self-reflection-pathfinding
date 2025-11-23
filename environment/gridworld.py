import numpy as np
import random
from collections import deque

class GridWorld:
    ACTION_UP = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_REFLECT = 4  # reflection action

    def __init__(self, size=5, reflect_cost=0.0, min_distance=4):
        self.size = size
        self.reflect_cost = reflect_cost
        self.min_distance = min_distance

        self.grid = np.zeros((size, size))
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []

    # ---------------------
    # Utility methods
    # ---------------------

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

    # ---------------------
    # Environment generation
    # ---------------------

    def set_agent_goal(self):
        positions = [(i,j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(positions)

        for a in positions:
            valid_goals = [
                g for g in positions if self.manhattan_distance(a, g) >= self.min_distance
            ]
            if not valid_goals:
                continue

            random.shuffle(valid_goals)
            for g in valid_goals:
                if self.is_path_exists(a, g):
                    self.agent_pos = a
                    self.goal_pos = g
                    return True
        return False

    def generate_obstacles(self, num_obstacles):
        self.obstacles = []
        free_positions = [
            (i,j)
            for i in range(self.size)
            for j in range(self.size)
            if (i,j) != self.agent_pos and (i,j) != self.goal_pos
        ]
        random.shuffle(free_positions)

        for pos in free_positions[:num_obstacles]:
            self.grid[pos] = 1
            self.obstacles.append(pos)

    def generate_environment(self, num_obstacles=3):
        self.grid = np.zeros((self.size, self.size))

        if not self.set_agent_goal():
            raise Exception("Could not place agent and goal")

        self.generate_obstacles(num_obstacles)

        if not self.is_path_exists(self.agent_pos, self.goal_pos):
            self.grid = np.zeros((self.size, self.size))
            self.obstacles = []
            self.generate_obstacles(max(0, num_obstacles - 1))

    # ---------------------
    # Environment API
    # ---------------------

    def reset(self):
        return self.agent_pos

    def step(self, action):
        old_pos = self.agent_pos
        old_dist = self.manhattan_distance(old_pos, self.goal_pos)

        # reflection action
        if action == self.ACTION_REFLECT:
            # Reflection cost only
            return old_pos, -0.05 - self.reflect_cost, False

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
            return old_pos, -0.3, False

        # update position
        self.agent_pos = (nx, ny)
        new_dist = self.manhattan_distance(self.agent_pos, self.goal_pos)

        # goal reward
        if self.agent_pos == self.goal_pos:
            return self.agent_pos, 5.0, True

        # shaped reward: moving closer or farther
        if new_dist < old_dist:
            reward = +0.1
        elif new_dist > old_dist:
            reward = -0.1
        else:
            reward = -0.05  # neutral step cost

        return self.agent_pos, reward, False
