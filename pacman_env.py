import numpy as np
import gym
from gym import spaces

class PacManEnv(gym.Env):
    """Improved Custom Pac-Man Environment for OpenAI Gym"""
    
    def __init__(self, grid_size=15):
        super(PacManEnv, self).__init__()
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        
        # Observation space: grid + ghost positions + additional features
        obs_size = grid_size * grid_size + 8 + 4  # grid + ghosts + distances
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(obs_size,), dtype=np.float32
        )
        
        # Game components
        self.pacman_pos = None
        self.ghosts = []
        self.pellets = []
        self.walls = []
        self.score = 0
        self.steps_taken = 0
        self.max_steps = 500  # Reduced for faster episodes
        self.pellets_collected = 0
        self.last_pellet_step = 0
        
        self.reset()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.pacman_pos = [self.grid_size // 2, self.grid_size // 2]
        
        # Initialize ghosts at corners (farther from pacman)
        self.ghosts = [
            [1, 1],
            [self.grid_size - 2, 1],
            [1, self.grid_size - 2],
            [self.grid_size - 2, self.grid_size - 2]
        ]
        
        # Initialize maze
        self._initialize_maze()
        
        self.score = 0
        self.steps_taken = 0
        self.pellets_collected = 0
        self.last_pellet_step = 0
        self.initial_pellets = len(self.pellets)
        
        return self._get_state()
    
    def _initialize_maze(self):
        """Create walls and pellets with simpler layout"""
        self.walls = []
        self.pellets = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Border walls
                if i == 0 or i == self.grid_size - 1 or j == 0 or j == self.grid_size - 1:
                    self.walls.append([i, j])
                # Simpler internal walls for easier navigation
                elif (i % 4 == 0 and j % 4 == 0 and i > 2 and i < self.grid_size - 3):
                    self.walls.append([i, j])
                # More pellets for positive reinforcement
                elif not self._is_ghost_position(i, j) and [i, j] != self.pacman_pos:
                    self.pellets.append([i, j])
    
    def _is_ghost_position(self, x, y):
        """Check if position has a ghost"""
        return any(g[0] == x and g[1] == y for g in self.ghosts)
    
    def _is_wall(self, x, y):
        """Check if position is a wall"""
        return any(w[0] == x and w[1] == y for w in self.walls)
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_state(self):
        """Get current state representation with additional features"""
        state = np.zeros(self.grid_size * self.grid_size + 8 + 4, dtype=np.float32)
        
        # Encode grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if self._is_wall(i, j):
                    state[idx] = -1
                elif any(p[0] == i and p[1] == j for p in self.pellets):
                    state[idx] = 1
                elif i == self.pacman_pos[0] and j == self.pacman_pos[1]:
                    state[idx] = 2
        
        # Encode ghost positions (normalized)
        base_idx = self.grid_size * self.grid_size
        for idx, ghost in enumerate(self.ghosts):
            state[base_idx + idx * 2] = ghost[0] / self.grid_size
            state[base_idx + idx * 2 + 1] = ghost[1] / self.grid_size
        
        # Add distance features to nearest ghost and pellet
        ghost_distances = [self._manhattan_distance(self.pacman_pos, g) for g in self.ghosts]
        min_ghost_dist = min(ghost_distances) if ghost_distances else 0
        
        pellet_distances = [self._manhattan_distance(self.pacman_pos, p) for p in self.pellets]
        min_pellet_dist = min(pellet_distances) if pellet_distances else 0
        
        feature_idx = base_idx + 8
        state[feature_idx] = min_ghost_dist / self.grid_size  # Normalized min ghost distance
        state[feature_idx + 1] = min_pellet_dist / self.grid_size  # Normalized min pellet distance
        state[feature_idx + 2] = len(self.pellets) / self.initial_pellets  # Pellets remaining ratio
        state[feature_idx + 3] = self.pellets_collected / self.initial_pellets  # Progress ratio
        
        return state
    
    def step(self, action):
        """Execute one step in the environment"""
        self.steps_taken += 1
        reward = 0  # No step penalty to encourage exploration
        done = False
        
        # Move Pac-Man
        new_pos = self.pacman_pos.copy()
        if action == 0:    # Up
            new_pos[1] -= 1
        elif action == 1:  # Down
            new_pos[1] += 1
        elif action == 2:  # Left
            new_pos[0] -= 1
        elif action == 3:  # Right
            new_pos[0] += 1
        
        # Check collision with walls
        if not self._is_wall(new_pos[0], new_pos[1]):
            self.pacman_pos = new_pos
            
            # Check pellet collection
            for i, pellet in enumerate(self.pellets):
                if pellet[0] == new_pos[0] and pellet[1] == new_pos[1]:
                    self.pellets.pop(i)
                    self.pellets_collected += 1
                    self.last_pellet_step = self.steps_taken
                    
                    # Progressive rewards - more reward as you collect more
                    base_reward = 10
                    progress_bonus = (self.pellets_collected / self.initial_pellets) * 20
                    reward = base_reward + progress_bonus
                    self.score += 10
                    break
            
            # Check ghost collision
            if self._is_ghost_position(new_pos[0], new_pos[1]):
                reward = -50  # Reduced penalty to encourage risk-taking
                done = True
            
            # Win condition with big reward
            if len(self.pellets) == 0:
                reward = 200  # Huge reward for winning
                done = True
        else:
            reward = -0.5  # Small wall collision penalty
        
        # Penalty for taking too long without collecting pellets
        if self.steps_taken - self.last_pellet_step > 50:
            reward -= 0.5
        
        # Move ghosts (less aggressively)
        if self.steps_taken % 2 == 0:  # Ghosts move every other step
            self._move_ghosts()
        
        # Check collision after ghost movement
        if self._is_ghost_position(self.pacman_pos[0], self.pacman_pos[1]):
            reward = -50
            done = True
        
        # Max steps termination
        if self.steps_taken >= self.max_steps:
            done = True
            # Reward based on progress
            progress_reward = (self.pellets_collected / self.initial_pellets) * 50
            reward += progress_reward
        
        state = self._get_state()
        info = {
            'score': self.score, 
            'steps': self.steps_taken,
            'pellets_collected': self.pellets_collected,
            'pellets_remaining': len(self.pellets)
        }
        
        return state, reward, done, info
    
    def _move_ghosts(self):
        """Move ghosts towards Pac-Man with some randomness"""
        for i, ghost in enumerate(self.ghosts):
            # 70% chase, 30% random
            if np.random.random() < 0.7:
                # Get valid moves
                possible_moves = [
                    [ghost[0], ghost[1] - 1],  # Up
                    [ghost[0], ghost[1] + 1],  # Down
                    [ghost[0] - 1, ghost[1]],  # Left
                    [ghost[0] + 1, ghost[1]]   # Right
                ]
                
                # Filter out walls
                valid_moves = [m for m in possible_moves if not self._is_wall(m[0], m[1])]
                
                if valid_moves:
                    # Move towards Pac-Man (Manhattan distance)
                    distances = [
                        abs(m[0] - self.pacman_pos[0]) + abs(m[1] - self.pacman_pos[1])
                        for m in valid_moves
                    ]
                    min_dist = min(distances)
                    best_moves = [m for m, d in zip(valid_moves, distances) if d == min_dist]
                    
                    # Randomly select among best moves
                    self.ghosts[i] = best_moves[np.random.randint(len(best_moves))]
            # else: ghost stays in place (30% chance)
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            grid = np.full((self.grid_size, self.grid_size), ' ', dtype=str)
            
            # Place walls
            for wall in self.walls:
                grid[wall[1], wall[0]] = '█'
            
            # Place pellets
            for pellet in self.pellets:
                grid[pellet[1], pellet[0]] = '·'
            
            # Place ghosts
            for ghost in self.ghosts:
                grid[ghost[1], ghost[0]] = 'G'
            
            # Place Pac-Man
            grid[self.pacman_pos[1], self.pacman_pos[0]] = 'P'
            
            # Print grid
            print('\n' + '─' * (self.grid_size * 2 + 1))
            for row in grid:
                print('│' + ' '.join(row) + '│')
            print('─' * (self.grid_size * 2 + 1))
            print(f'Score: {self.score} | Steps: {self.steps_taken} | '
                  f'Collected: {self.pellets_collected}/{self.initial_pellets} | '
                  f'Remaining: {len(self.pellets)}')
    
    def close(self):
        """Clean up resources"""
        pass