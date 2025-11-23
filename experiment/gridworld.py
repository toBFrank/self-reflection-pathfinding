import numpy as np
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

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = []

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

    def generate_environment(self, num_obstacles=3):
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
