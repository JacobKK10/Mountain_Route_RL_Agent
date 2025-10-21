import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MountainRouteEnv(gym.Env):

    metadata = {"render_mode": ["human"], "render_fps": 30}
    def __init__(self, render_mode=None):
        
        super().__init__()

        #array [position, velocity]
        self.obs_space = spaces.Box(low=np.array([0.0,-1.0]), high=np.array([1.0,1.0]), dtype=np.float32)
        #possible actions -> 0 - slow down, 1 - stay, 2 - accelerate
        self.act_space = spaces.Discrete(3)
        #begging position [0, 0]
        self.start_state = np.array([0.0,0.0], dtype=np.float32)
        #ending position 1
        self.end_state = 1.0
        #obstacles on route
        self.obstacles = [0.3, 0.5, 0.8]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0,0.0], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        position, velocity = self.state

        if action == 0:
            velocity -= 0.01
        elif action == 2:
            velocity += 0.01

        velocity = np.clip(velocity, -0.05, 0.05)
        position += velocity
        position = np.clip(position, 0.0, 1.0)

        #penalty to end each episode faster
        reward = -0.01

        for obstacle in self.obstacles:
            if abs(position - obstacle) < 0.02:
                reward -= 1.0
                velocity = 0.0

        done = False
        if position >= self.end_state:
            #Reward for reaching the end
            reward += 10.0
            done = True

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, done, False, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600,200))
            pygame.display.set_caption("Mountain Route")
            self.clock = pygame.time.Clock()

        self.screen.fill((135,206,235))
        pygame.draw.rect(self.screen, (34,139,34), (0,150,600,50))
        pygame.draw.line(self.screen, (50,50,50), (50,150), (550,150), 5)

        car = int(50 + self.state[0] * 500)
        pygame.draw.rect(self.screen, (255, 0, 0), (car - 10, 130, 20, 10))
        
        for obstacle_2 in self.obstacles:
            obstacle_x = int(50 + obstacle_2 * 500)
            obstacle_y = 145
            pygame.draw.circle(self.screen, (80, 80, 80), (obstacle_x, obstacle_y), 6)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None