import pygame
import numpy as np
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
import sys
import time

# Colors
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
PINK = (255, 182, 193)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

class PacManVisualizer:
    """Pygame visualizer for trained Pac-Man agent"""
    
    def __init__(self, agent, env, cell_size=40):
        pygame.init()
        
        self.agent = agent
        self.env = env
        self.cell_size = cell_size
        self.grid_size = env.grid_size
        
        # Window dimensions
        self.info_width = 300
        self.grid_width = self.grid_size * cell_size
        self.grid_height = self.grid_size * cell_size
        self.window_width = self.grid_width + self.info_width
        self.window_height = self.grid_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('DQN Pac-Man Agent')
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Animation variables
        self.pacman_mouth_angle = 45
        self.pacman_mouth_opening = True
        self.ghost_animation_offset = 0
        
        # Game statistics
        self.episode = 0
        self.total_score = 0
        self.episodes_history = []
        
    def draw_grid(self):
        """Draw the game grid"""
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, GRAY, 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.grid_height), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, GRAY, 
                           (0, i * self.cell_size), 
                           (self.grid_width, i * self.cell_size), 1)
    
    def draw_walls(self):
        """Draw maze walls"""
        for wall in self.env.walls:
            x = wall[0] * self.cell_size
            y = wall[1] * self.cell_size
            pygame.draw.rect(self.screen, BLUE, 
                           (x, y, self.cell_size, self.cell_size))
            # Add brick pattern
            pygame.draw.rect(self.screen, CYAN, 
                           (x, y, self.cell_size, self.cell_size), 2)
    
    def draw_pellets(self):
        """Draw pellets"""
        for pellet in self.env.pellets:
            x = pellet[0] * self.cell_size + self.cell_size // 2
            y = pellet[1] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, WHITE, (x, y), 5)
    
    def draw_pacman(self):
        """Draw animated Pac-Man"""
        x = self.env.pacman_pos[0] * self.cell_size + self.cell_size // 2
        y = self.env.pacman_pos[1] * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 5
        
        # Animate mouth
        if self.pacman_mouth_opening:
            self.pacman_mouth_angle += 3
            if self.pacman_mouth_angle >= 45:
                self.pacman_mouth_opening = False
        else:
            self.pacman_mouth_angle -= 3
            if self.pacman_mouth_angle <= 5:
                self.pacman_mouth_opening = True
        
        # Draw Pac-Man body
        start_angle = np.radians(self.pacman_mouth_angle)
        end_angle = np.radians(360 - self.pacman_mouth_angle)
        
        # Create points for the pie slice
        points = [(x, y)]
        for angle in np.linspace(start_angle, end_angle, 30):
            px = x + int(radius * np.cos(angle))
            py = y + int(radius * np.sin(angle))
            points.append((px, py))
        points.append((x, y))
        
        pygame.draw.polygon(self.screen, YELLOW, points)
        
        # Draw eye
        eye_x = x + radius // 3
        eye_y = y - radius // 3
        pygame.draw.circle(self.screen, BLACK, (eye_x, eye_y), 3)
    
    def draw_ghosts(self):
        """Draw animated ghosts"""
        ghost_colors = [RED, PINK, ORANGE, CYAN]
        
        for i, ghost in enumerate(self.env.ghosts):
            x = ghost[0] * self.cell_size + self.cell_size // 2
            y = ghost[1] * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 5
            
            color = ghost_colors[i % len(ghost_colors)]
            
            # Draw ghost body (circle + rectangle)
            pygame.draw.circle(self.screen, color, (x, y - 3), radius)
            pygame.draw.rect(self.screen, color, 
                           (x - radius, y - 3, radius * 2, radius + 3))
            
            # Draw wavy bottom with animation
            wave_points = []
            num_waves = 4
            for j in range(num_waves + 1):
                wave_x = x - radius + (j * radius * 2 // num_waves)
                wave_y = y + radius + int(3 * np.sin(self.ghost_animation_offset + j))
                wave_points.append((wave_x, wave_y))
            
            # Close the shape
            wave_points.append((x + radius, y))
            wave_points.append((x - radius, y))
            
            pygame.draw.polygon(self.screen, color, wave_points)
            
            # Draw eyes
            eye_radius = 4
            left_eye_x = x - radius // 3
            right_eye_x = x + radius // 3
            eye_y = y - 5
            
            # White part of eyes
            pygame.draw.circle(self.screen, WHITE, (left_eye_x, eye_y), eye_radius)
            pygame.draw.circle(self.screen, WHITE, (right_eye_x, eye_y), eye_radius)
            
            # Black pupils (looking at Pac-Man)
            pacman_x = self.env.pacman_pos[0] * self.cell_size + self.cell_size // 2
            pacman_y = self.env.pacman_pos[1] * self.cell_size + self.cell_size // 2
            
            dx = pacman_x - x
            dy = pacman_y - y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                pupil_offset = 2
                pupil_dx = int((dx / distance) * pupil_offset)
                pupil_dy = int((dy / distance) * pupil_offset)
            else:
                pupil_dx = pupil_dy = 0
            
            pygame.draw.circle(self.screen, BLUE, 
                             (left_eye_x + pupil_dx, eye_y + pupil_dy), 2)
            pygame.draw.circle(self.screen, BLUE, 
                             (right_eye_x + pupil_dx, eye_y + pupil_dy), 2)
        
        # Update animation
        self.ghost_animation_offset += 0.3
    
    def draw_info_panel(self):
        """Draw information panel"""
        panel_x = self.grid_width
        panel_rect = pygame.Rect(panel_x, 0, self.info_width, self.window_height)
        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        
        # Title
        title_text = self.font_medium.render("DQN Agent", True, YELLOW)
        self.screen.blit(title_text, (panel_x + 20, 20))
        
        # Episode
        y_offset = 80
        episode_text = self.font_small.render(f"Episode: {self.episode}", True, WHITE)
        self.screen.blit(episode_text, (panel_x + 20, y_offset))
        
        # Score
        y_offset += 50
        score_text = self.font_small.render(f"Score: {self.env.score}", True, GREEN)
        self.screen.blit(score_text, (panel_x + 20, y_offset))
        
        # Steps
        y_offset += 50
        steps_text = self.font_small.render(f"Steps: {self.env.steps_taken}", True, WHITE)
        self.screen.blit(steps_text, (panel_x + 20, y_offset))
        
        # Pellets remaining
        y_offset += 50
        pellets_text = self.font_small.render(f"Pellets: {len(self.env.pellets)}", True, WHITE)
        self.screen.blit(pellets_text, (panel_x + 20, y_offset))
        
        # Epsilon
        y_offset += 50
        epsilon_text = self.font_small.render(f"Epsilon: {self.agent.epsilon:.3f}", True, CYAN)
        self.screen.blit(epsilon_text, (panel_x + 20, y_offset))
        
        # Average score
        if self.episodes_history:
            y_offset += 50
            avg_score = np.mean(self.episodes_history[-100:])
            avg_text = self.font_small.render(f"Avg Score: {avg_score:.1f}", True, ORANGE)
            self.screen.blit(avg_text, (panel_x + 20, y_offset))
        
        # Controls
        y_offset = self.window_height - 150
        controls_title = self.font_small.render("Controls:", True, YELLOW)
        self.screen.blit(controls_title, (panel_x + 20, y_offset))
        
        y_offset += 40
        controls = [
            "SPACE: Pause",
            "R: Restart",
            "Q: Quit"
        ]
        for control in controls:
            control_text = self.font_small.render(control, True, WHITE)
            self.screen.blit(control_text, (panel_x + 20, y_offset))
            y_offset += 30
    
    def draw_game_over(self, win=False):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.grid_width, self.grid_height))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        if win:
            text = self.font_large.render("YOU WIN!", True, GREEN)
        else:
            text = self.font_large.render("GAME OVER", True, RED)
        
        text_rect = text.get_rect(center=(self.grid_width // 2, self.grid_height // 2 - 50))
        self.screen.blit(text, text_rect)
        
        # Score text
        score_text = self.font_medium.render(f"Score: {self.env.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.grid_width // 2, self.grid_height // 2 + 20))
        self.screen.blit(score_text, score_rect)
        
        # Restart text
        restart_text = self.font_small.render("Press R to restart", True, WHITE)
        restart_rect = restart_text.get_rect(center=(self.grid_width // 2, self.grid_height // 2 + 80))
        self.screen.blit(restart_text, restart_rect)
    
    def run_episode(self, fps=10, pause_on_done=True):
        """Run one episode with visualization"""
        state = self.env.reset()
        done = False
        paused = False
        
        self.episode += 1
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return False
                    if event.key == pygame.K_r:
                        return True
                    if event.key == pygame.K_SPACE:
                        paused = not paused
            
            if not paused and not done:
                # Agent selects action
                action = self.agent.act(state)
                state, reward, done, info = self.env.step(action)
            
            # Draw everything
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_walls()
            self.draw_pellets()
            self.draw_ghosts()
            self.draw_pacman()
            self.draw_info_panel()
            
            if done:
                win = len(self.env.pellets) == 0
                self.draw_game_over(win)
                self.episodes_history.append(self.env.score)
                
                if pause_on_done:
                    # Wait for user input
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_q:
                                    return False
                                if event.key == pygame.K_r:
                                    return True
                        pygame.display.flip()
                        self.clock.tick(30)
                else:
                    pygame.display.flip()
                    time.sleep(2)
                    return True
            
            pygame.display.flip()
            self.clock.tick(fps)
    
    def run(self, num_episodes=10, fps=10, pause_on_done=True):
        """Run multiple episodes"""
        for _ in range(num_episodes):
            continue_running = self.run_episode(fps, pause_on_done)
            if not continue_running:
                break
        
        pygame.quit()

def visualize_agent(model_path='models/dqn_pacman_final.pth', 
                   num_episodes=10, 
                   fps=10,
                   grid_size=15,
                   pause_on_done=True):
    """
    Visualize a trained DQN agent playing Pac-Man
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to run
        fps: Frames per second (game speed)
        grid_size: Size of the game grid
        pause_on_done: Whether to pause after each episode
    """
    # Initialize environment and agent
    env = PacManEnv(grid_size=grid_size)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Epsilon: {agent.epsilon}")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Using untrained agent...")
    
    # Set epsilon to 0 for pure exploitation (no exploration)
    agent.epsilon = 0.0
    
    # Create visualizer and run
    visualizer = PacManVisualizer(agent, env, cell_size=40)
    print("\nStarting visualization...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  R: Restart episode")
    print("  Q: Quit")
    print("-" * 60)
    
    visualizer.run(num_episodes=num_episodes, fps=fps, pause_on_done=pause_on_done)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DQN Pac-Man Agent')
    parser.add_argument('--model', type=str, default='models/dqn_pacman_final.pth',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (game speed)')
    parser.add_argument('--grid-size', type=int, default=15,
                       help='Size of game grid')
    parser.add_argument('--no-pause', action='store_true',
                       help='Don\'t pause after each episode')
    
    args = parser.parse_args()
    
    visualize_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        grid_size=args.grid_size,
        pause_on_done=not args.no_pause
    )