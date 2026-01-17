"""
Adversarial Curriculum Training for Smart AGI Brain

This script trains the brain using the Experiencer vs Saboteur paradigm:
- Experiencer tries to maximize score
- Saboteur tries to create challenging scenarios
- Both share the SAME brain, forcing genuine understanding

Key Metrics to Watch:
1. Score improvement over time
2. Modularity changes (should increase as skills cluster)
3. Hebbian team formation
4. Growth/pruning dynamics
5. Experiencer vs Saboteur balance (neither should dominate)
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json

from smart_agi.smart_brain import SmartBrain, SmartBrainConfig
from smart_agi.envs.flappy_bird import FlappyBirdEnv
from smart_agi.training.adversarial import (
    AdversarialCurriculum, AdversarialConfig,
    FlappyBirdChallengeGenerator, Role
)
from smart_agi.core.mirror import MirrorMode


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_generations: int = 20
    episodes_per_generation: int = 10
    max_steps: int = 500
    experiencer_episodes: int = 5
    saboteur_episodes: int = 5
    verbose: bool = True


def train_with_adversarial_curriculum(config: TrainingConfig) -> Tuple[SmartBrain, Dict]:
    """
    Train brain with adversarial curriculum.
    
    Returns:
        Tuple of (trained_brain, training_history)
    """
    # Create brain
    brain_config = SmartBrainConfig(
        observation_dim=8,
        action_dim=2,
        sheet_size=4,
        hidden_dim=32,
        embedding_dim=32,
        top_k=4,
        num_liquids=4
    )
    brain = SmartBrain(brain_config)
    
    # Create adversarial curriculum
    adv_config = AdversarialConfig(
        experiencer_steps=10,
        saboteur_steps=5,
        target_success_rate=0.5,
        difficulty_adjustment_rate=0.1,
        min_difficulty=0.1,
        max_difficulty=2.0
    )
    challenge_gen = FlappyBirdChallengeGenerator()
    curriculum = AdversarialCurriculum(adv_config, challenge_gen)
    
    # Create environment
    env = FlappyBirdEnv()
    
    # Training history
    history = {
        'generations': [],
        'experiencer_scores': [],
        'saboteur_scores': [],
        'modularity': [],
        'difficulty': [],
        'mirror_consistency': []
    }
    
    for gen in range(config.num_generations):
        gen_results = {
            'experiencer': {'scores': [], 'rewards': []},
            'saboteur': {'scores': [], 'rewards': []}
        }
        
        # === Experiencer Phase ===
        brain.set_role(Role.EXPERIENCER)
        challenge = curriculum.generate_challenge({})
        
        for ep in range(config.experiencer_episodes):
            obs = env.reset()
            
            # Apply challenge modifications
            if 'gravity_mult' in challenge.parameters:
                env.gravity = 0.5 * challenge.parameters['gravity_mult']
            if 'gap_size' in challenge.parameters:
                env.gap_size = int(150 * challenge.parameters['gap_size'])
            
            brain.reset()
            episode_reward = 0.0
            
            for step in range(config.max_steps):
                action_logits, info = brain.forward(obs)
                action = brain.get_action(action_logits)
                
                next_obs, reward, done, _ = env.step(action)
                episode_reward += reward
                
                brain.learn(obs, action, next_obs, reward, done)
                obs = next_obs
                
                if done:
                    break
            
            gen_results['experiencer']['scores'].append(env.score)
            gen_results['experiencer']['rewards'].append(episode_reward)
        
        # Record experiencer performance
        exp_success = np.mean(gen_results['experiencer']['scores']) > 0
        curriculum.record_solve_attempt(
            challenge.challenge_id, exp_success, 
            np.mean(gen_results['experiencer']['rewards']),
            config.max_steps, 'experiencer'
        )
        
        # === Saboteur Phase ===
        brain.set_role(Role.SABOTEUR)
        
        for ep in range(config.saboteur_episodes):
            obs = env.reset()
            brain.reset()
            episode_reward = 0.0
            
            for step in range(config.max_steps):
                action_logits, info = brain.forward(obs)
                
                # Saboteur: try to find actions that lead to failure
                # Use inverse model to plan toward "bad" states
                goal_state = obs.copy()
                goal_state[1] = -1.0  # Try to go down (crash)
                
                planned_action = brain.mirror.compute_action(obs, goal_state)
                action = int(planned_action[0] > 0.5)  # Threshold to discrete
                
                next_obs, reward, done, _ = env.step(action)
                
                # Saboteur reward: inverse of experiencer reward
                saboteur_reward = -reward if reward > 0 else 0.1
                episode_reward += saboteur_reward
                
                brain.learn(obs, action, next_obs, saboteur_reward, done)
                obs = next_obs
                
                if done:
                    break
            
            gen_results['saboteur']['scores'].append(env.score)
            gen_results['saboteur']['rewards'].append(episode_reward)
        
        # Record saboteur performance (saboteur succeeds when experiencer fails)
        sab_success = np.mean(gen_results['saboteur']['scores']) < 1
        # Difficulty is auto-adjusted in record_solve_attempt
        
        # Compute mirror consistency
        test_obs = np.random.randn(8).astype(np.float32)
        test_action = np.array([0.5, 0.5], dtype=np.float32)
        pred_next = brain.mirror.predict_next_state(test_obs, test_action)
        recovered_action = brain.mirror.compute_action(test_obs, pred_next)
        mirror_consistency = 1.0 - np.mean((test_action - recovered_action) ** 2)
        
        # Record history
        history['generations'].append(gen)
        history['experiencer_scores'].append(np.mean(gen_results['experiencer']['scores']))
        history['saboteur_scores'].append(np.mean(gen_results['saboteur']['scores']))
        history['modularity'].append(brain.sheet.compute_modularity())
        history['difficulty'].append(curriculum.current_difficulty)
        history['mirror_consistency'].append(mirror_consistency)
        
        if config.verbose:
            print(f"Gen {gen:3d}: "
                  f"Exp={np.mean(gen_results['experiencer']['scores']):.1f}, "
                  f"Sab={np.mean(gen_results['saboteur']['scores']):.1f}, "
                  f"Mod={history['modularity'][-1]:.3f}, "
                  f"Diff={curriculum.current_difficulty:.2f}, "
                  f"Mirror={mirror_consistency:.3f}")
    
    return brain, history


def analyze_results(brain: SmartBrain, history: Dict) -> Dict[str, Any]:
    """Analyze training results for self-organization indicators."""
    
    analysis = {
        'performance': {
            'final_exp_score': history['experiencer_scores'][-1],
            'max_exp_score': max(history['experiencer_scores']),
            'score_improvement': history['experiencer_scores'][-1] - history['experiencer_scores'][0]
        },
        'self_organization': {
            'modularity_change': history['modularity'][-1] - history['modularity'][0],
            'final_modularity': history['modularity'][-1],
            'modularity_trend': 'increasing' if history['modularity'][-1] > history['modularity'][0] else 'decreasing'
        },
        'adversarial_balance': {
            'exp_sab_ratio': np.mean(history['experiencer_scores']) / (np.mean(history['saboteur_scores']) + 0.1),
            'difficulty_range': (min(history['difficulty']), max(history['difficulty']))
        },
        'mirror_model': {
            'final_consistency': history['mirror_consistency'][-1],
            'avg_consistency': np.mean(history['mirror_consistency'])
        },
        'brain_stats': brain.get_stats()
    }
    
    return analysis


if __name__ == "__main__":
    print("=== Adversarial Curriculum Training ===\n")
    
    config = TrainingConfig(
        num_generations=30,
        episodes_per_generation=10,
        max_steps=300,
        experiencer_episodes=5,
        saboteur_episodes=5,
        verbose=True
    )
    
    brain, history = train_with_adversarial_curriculum(config)
    
    print("\n=== Training Complete ===")
    analysis = analyze_results(brain, history)
    
    print("\nPerformance:")
    for k, v in analysis['performance'].items():
        print(f"  {k}: {v}")
    
    print("\nSelf-Organization:")
    for k, v in analysis['self_organization'].items():
        print(f"  {k}: {v}")
    
    print("\nAdversarial Balance:")
    for k, v in analysis['adversarial_balance'].items():
        print(f"  {k}: {v}")
    
    print("\nMirror Model:")
    for k, v in analysis['mirror_model'].items():
        print(f"  {k}: {v}")
    
    print(f"\nFinal Brain: {brain}")
