import numpy as np
import gym
import gym.utils.seeding as seeding
from typing import Tuple

# T = Tile displaying the color of the button to press
# A = Position of the Agent
# B = Button
MAPS = {
    'easy': [
        ".T.",
        ".A.",
        "...",
        "B.B",
    ]
}

EMPTY_CHAR = b'.'
WALL_CHAR = b'W'
TILE_CHAR = b'T'
AGENT_CHAR = b'A'
BUTTON_CHAR = b'B'

BUTTON_SYMBOLS = "BRYGP"
TILE_SYMBOLS = "brygp"

class PressTheButtonEnv(gym.Env):
    def __init__(self, gridworld, view_range=1):
        if isinstance(gridworld, str):
            assert gridworld in MAPS, f"Unknown map name '{gridworld}'"
            gridworld = MAPS[gridworld]
        
        self.view_range = view_range
        self._parse_map(gridworld)
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.view_range * 2 + 1, self.view_range * 2 + 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        
        self.seed()
        self.reset()
        
    def _parse_map(self, gridworld):
        self._gridworld = np.array(gridworld, dtype='c')
        self._gridworld = np.pad(self._gridworld, self.view_range, constant_values='W')
        self.height, self.width = self._gridworld.shape
        
        # Find agent position and erase it from the gridworld
        agent_positions = np.argwhere(self._gridworld == AGENT_CHAR)
        assert len(agent_positions) == 1, "Found less or more than 1 agent in the gridworld."
        self._initial_agent_pos = (*agent_positions[0],)
        self._gridworld[self._initial_agent_pos] = EMPTY_CHAR
        
        # Find the tile position
        tile_positions = np.argwhere(self._gridworld == TILE_CHAR)
        assert len(tile_positions) == 1, "Found less or more than 1 coloured tile in the gridworld."
        self._tile_position = (*tile_positions[0],)
        
        # Find the button positions
        self._button_positions = [(*pos,) for pos in np.argwhere(self._gridworld == BUTTON_CHAR)]
        
    def reset(self):
        # Reset position
        self._agent_x, self._agent_y = self._initial_agent_pos
        
        # Choose a random button
        self._button_idx = self.np_random.randint(len(self._button_positions))
        
        # Assign colors
        randomized_indices = np.arange(len(BUTTON_SYMBOLS))
        self.np_random.shuffle(randomized_indices)
        for i, button_pos in enumerate(self._button_positions):
            self._gridworld[button_pos] = BUTTON_SYMBOLS[randomized_indices[i]]
            if i == self._button_idx:
                self._gridworld[self._tile_position] = TILE_SYMBOLS[randomized_indices[i]]
        
        return self._get_observation()
        
    def step(self, action):
        if action == 0: # UP
            new_x, new_y = self._agent_x, max(0, self._agent_y - 1)
        elif action == 1: # DOWN
            new_x, new_y = self._agent_x, min(self.height, self._agent_y + 1)
        elif action == 2: # LEFT
            new_x, new_y = max(0, self._agent_x - 1), self._agent_y
        elif action == 3: # RIGHT
            new_x, new_y = min(self.width, self._agent_x + 1), self._agent_y
        else:
            raise Exception("Invalid action")
        
        # Move the agent
        if self._gridworld[new_y, new_x] != WALL_CHAR:
            self._agent_x, self._agent_y = new_x, new_y
            
        # Compute reward and check if done
        done = False
        reward = -1
        for i, button_pos in enumerate(self._button_positions):
            if (self._agent_y, self._agent_x) == button_pos:
                done = True
                reward = 10 if i == self._button_idx else -10
                break
            
        return self._get_observation(), reward, done, None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self):
        for y, row in enumerate(self._gridworld):
            for x, c in enumerate(row):
                if (y, x) == (self._agent_y, self._agent_x):
                    print('A', end='')
                else:
                    print(c.decode("utf-8"), end='')
            print()
        print()
    
    def _get_observation(self):
        y_slice = slice(self._agent_y - self.view_range, self._agent_y + self.view_range + 1)
        x_slice = slice(self._agent_x - self.view_range, self._agent_x + self.view_range + 1)
        return self._gridworld[y_slice, x_slice].view('uint8')
