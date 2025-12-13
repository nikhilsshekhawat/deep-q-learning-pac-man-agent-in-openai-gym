import numpy as np
import matplotlib.pyplot as plt
from pacman_env import PacManEnv
from dqn_agent import DQNAgent
import time

def train_dqn(episodes=2000, render_frequency=100, save_frequency=200):
    """Train DQN agent on Pac-Man environment"""
    
    # Initialize environment and agent
    env = PacManEnv(grid_size=15)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    pellets_collected = []
    win_count = 0
    
    print("Starting training...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Device: {agent.device}")
    print("-" * 60)
    
    best_avg_score = -float('inf')
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Select and perform action
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            agent.steps += 1
        
        # Track win
        if len(env.pellets) == 0:
            win_count += 1
        
        # Record metrics
        scores.append(info['score'])
        pellets_collected.append(env.pellets_collected)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_wins = sum(1 for i in range(max(0, episode-99), episode+1) 
                            if i < len(scores) and 
                            pellets_collected[i] == env.initial_pellets)
            win_rate = recent_wins / min(100, episode + 1) * 100
            
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {info['score']:6.1f} | "
                  f"Avg: {avg_score:6.1f} | "
                  f"Pellets: {env.pellets_collected}/{env.initial_pellets} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Wins: {win_count} ({win_rate:.1f}%)")
        
        # Save best model
        if avg_score > best_avg_score and episode > 100:
            best_avg_score = avg_score
            agent.save('models/dqn_pacman_best.pth')
        
        # Save model periodically
        if (episode + 1) % save_frequency == 0:
            agent.save(f'models/dqn_pacman_ep{episode+1}.pth')
            print(f"✓ Model saved at episode {episode + 1}")
        
        # Optional: render occasionally
        if episode % render_frequency == 0 and episode > 0:
            print(f"\n--- Episode {episode} Visualization ---")
            env.render()
    
    print("-" * 60)
    print("Training completed!")
    print(f"Total wins: {win_count}/{episodes} ({win_count/episodes*100:.1f}%)")
    print(f"Best average score: {best_avg_score:.2f}")
    
    # Plot results
    plot_training_results(scores, avg_scores, losses, epsilons, pellets_collected, env.initial_pellets)
    
    # Save final model
    agent.save('models/dqn_pacman_final.pth')
    print("Final model saved as 'dqn_pacman_final.pth'")
    
    return agent, scores, avg_scores

def plot_training_results(scores, avg_scores, losses, epsilons, pellets_collected, total_pellets):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    episodes = list(range(1, len(scores) + 1))
    
    # Scores
    axes[0, 0].plot(episodes, scores, alpha=0.3, label='Score', color='blue')
    axes[0, 0].plot(episodes, avg_scores, linewidth=2, label='Avg Score (100)', color='red')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Training Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(episodes, losses, color='orange', alpha=0.6)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Epsilon
    axes[0, 2].plot(episodes, epsilons, color='green')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].set_title('Exploration Rate (Epsilon)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Pellets collected
    axes[1, 0].plot(episodes, pellets_collected, alpha=0.5, color='purple')
    avg_pellets = [np.mean(pellets_collected[max(0,i-99):i+1]) for i in range(len(pellets_collected))]
    axes[1, 0].plot(episodes, avg_pellets, linewidth=2, color='darkviolet', label='Avg (100)')
    axes[1, 0].axhline(y=total_pellets, color='red', linestyle='--', label=f'Max ({total_pellets})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Pellets Collected')
    axes[1, 0].set_title('Pellets Collected per Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Win rate (rolling 100 episodes)
    win_rate = []
    for i in range(len(pellets_collected)):
        start_idx = max(0, i - 99)
        wins = sum(1 for p in pellets_collected[start_idx:i+1] if p == total_pellets)
        win_rate.append(wins / min(100, i + 1) * 100)
    
    axes[1, 1].plot(episodes, win_rate, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].set_title('Win Rate (Rolling 100 Episodes)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1, 2].hist(scores, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Score')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Score Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Training plots saved as 'training_results.png'")
    plt.show()

def test_agent(agent, num_episodes=10, render=True):
    """Test trained agent"""
    env = PacManEnv(grid_size=15)
    test_scores = []
    wins = 0
    
    print("\nTesting agent...")
    print("-" * 60)
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if render and episode < 3:  # Only render first 3 episodes
                env.render()
                time.sleep(0.1)
        
        if len(env.pellets) == 0:
            wins += 1
            
        test_scores.append(info['score'])
        print(f"Test Episode {episode + 1}: Score = {info['score']}, "
              f"Pellets = {env.pellets_collected}/{env.initial_pellets}, "
              f"Steps = {info['steps']}, "
              f"Result = {'WIN' if len(env.pellets) == 0 else 'LOSS'}")
    
    agent.epsilon = original_epsilon
    
    print("-" * 60)
    print(f"Average Test Score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"Max Score: {max(test_scores)}")
    print(f"Min Score: {min(test_scores)}")
    print(f"Wins: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")
    
    return test_scores

if __name__ == "__main__":
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train agent with more episodes
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    agent, scores, avg_scores = train_dqn(episodes=2000, render_frequency=200, save_frequency=200)
    
    # Test agent with text rendering
    print("\n" + "=" * 60)
    print("TESTING PHASE (Text)")
    print("=" * 60)
    test_scores = test_agent(agent, num_episodes=10, render=False)
    
    # Launch Pygame visualization
    print("\n" + "=" * 60)
    print("LAUNCHING PYGAME VISUALIZATION")
    print("=" * 60)
    print("\nTo see the agent play with graphics, run:")
    print("  python visualize.py")
    print("\nOr use the best model:")
    print("  python visualize.py --model models/dqn_pacman_best.pth --episodes 10 --fps 8")