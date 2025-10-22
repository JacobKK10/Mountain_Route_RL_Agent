import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class MountainRouteChoiceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(low=0, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.min_length = 5
        self.max_length = 15
        self.max_obstacles = 10
        self.obstacle_penalty = 0.5
        self.length_penalty = 1.0

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.width, self.height = 600, 300

        self.state = None
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        self.length_A = np.random.randint(self.min_length, self.max_length)
        self.length_B = np.random.randint(self.min_length, self.max_length)
        self.obstacles_A = np.random.randint(0, self.max_obstacles)
        self.obstacles_B = np.random.randint(0, self.max_obstacles)

        self.state = np.array(
            [self.obstacles_A, self.obstacles_B, self.length_A, self.length_B],
            dtype=np.float32
        )

        self.last_action = None
        return self.state, {}

    def step(self, action):
        self.last_action = action

        if action == 0:
            time = self.length_A * self.length_penalty + self.obstacles_A * self.obstacle_penalty
        else:
            time = self.length_B * self.length_penalty + self.obstacles_B * self.obstacle_penalty

        reward = -time
        done = True
        return self.state, reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Mountain Route Choice")
            self.clock = pygame.time.Clock()

        self.screen.fill((200, 230, 255))

        pygame.draw.line(self.screen, (60, 60, 60), (50, 100), (550, 100), 4)
        pygame.draw.line(self.screen, (60, 60, 60), (50, 200), (550, 200), 4)

        random.seed(0)
        for _ in range(self.obstacles_A):
            x = random.randint(60, 540)
            pygame.draw.circle(self.screen, (180, 50, 50), (x, 100), 6)
        for _ in range(self.obstacles_B):
            x = random.randint(60, 540)
            pygame.draw.circle(self.screen, (180, 50, 50), (x, 200), 6)

        font = pygame.font.SysFont(None, 24)
        textA = font.render(f"Route A: length {self.length_A}, obstacles {self.obstacles_A}", True, (0, 0, 0))
        textB = font.render(f"Route B: length {self.length_B}, obstacles {self.obstacles_B}", True, (0, 0, 0))
        self.screen.blit(textA, (50, 50))
        self.screen.blit(textB, (50, 230))

        if self.last_action == 0:
            pygame.draw.rect(self.screen, (0, 255, 0), (45, 90, 520, 20), 2)
        elif self.last_action == 1:
            pygame.draw.rect(self.screen, (0, 255, 0), (45, 190, 520, 20), 2)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None