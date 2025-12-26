"""
Example: CartPole with Asymmetric TD Learning
==============================================
Demonstrates StableDQN on the CartPole environment.
"""

import gymnasium as gym
import numpy as np
from asymmetric_td import StableDQN

def train_cartpole():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create agent
    agent = StableDQN(
        state_dim=4,
        action_dim=2,
        gamma=0.99,
        lr=0.001,
        atd_weights=(0.5, 1.5),  # Asymmetric TD!
        epsilon_decay=0.995
    )
    
    print("Training CartPole with Asymmetric TD Learning...")
    print("-" * 50)
    
    scores = []
    
    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        agent.update_epsilon()
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode:4d} | Avg Score: {avg:6.1f} | Epsilon: {agent.epsilon:.3f}")
    
    print("-" * 50)
    print(f"Final Average: {np.mean(scores[-100:]):.1f}")
    
    env.close()

if __name__ == "__main__":
    train_cartpole()
