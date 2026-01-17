"""
Smart AGI Brain - Complete Integrated System

This module integrates ALL components into a unified AGI-like system:

1. BALANCED TERNARY FOUNDATION
   - Weights in {-1, 0, +1}
   - Activations in balanced quinary {-2, -1, 0, +1, +2}
   
2. SELF-ORGANIZING SHEET (GyrusNet)
   - 2D grid of microcolumns
   - Local Excitation / Lateral Inhibition (LE/LI)
   - Affinity liquids for skill clustering
   
3. HEBBIAN COUPLING
   - "Cells that fire together wire together"
   - Team formation through co-activation
   
4. DYNAMIC GROWTH/PRUNING
   - Grow when utilization is high
   - Prune when resources are wasted
   - Budget-constrained adaptation
   
5. TOOL USE
   - Math, Memory, Reasoning as first-class actions
   - Tool selection based on query similarity
   
6. MIRROR GENERALIZATION
   - Unified forward/inverse model
   - Experiencer (predict) and Saboteur (plan) share representation
   
7. ADVERSARIAL CURRICULUM
   - Self-play between Experiencer and Saboteur roles
   - Automatic difficulty calibration

The system is designed to exhibit emergent self-organization:
- Different agents evolve different neuron counts
- Clear affinity regions form on the sheet
- Hebbian links concentrate within modules
- Harder tasks push growth; easy tasks favor efficiency
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Import all components
from smart_agi.core.ternary import quantize_to_quinary
from smart_agi.core.neuron import NeuronLayer
from smart_agi.core.microcolumn import Microcolumn, MicrocolumnConfig, ColumnMode
from smart_agi.core.sheet import Sheet, SheetConfig
from smart_agi.core.router import Router, RouterConfig
from smart_agi.core.growth import GrowthController, GrowthConfig
from smart_agi.core.mirror import UnifiedMirrorModel, MirrorConfig, MirrorMode
from smart_agi.liquids.liquid import LiquidPool, LiquidConfig
from smart_agi.tools.tool_system import ToolRegistry, ToolSelector
from smart_agi.training.adversarial import (
    AdversarialCurriculum, AdversarialConfig,
    FlappyBirdChallengeGenerator, Role
)


@dataclass
class SmartBrainConfig:
    """Configuration for the complete Smart AGI Brain."""
    # Input/Output dimensions
    observation_dim: int = 8
    action_dim: int = 2
    
    # Sheet configuration
    sheet_size: int = 4  # 4x4 = 16 columns
    hidden_dim: int = 32
    embedding_dim: int = 32
    
    # Router
    top_k: int = 4
    
    # Liquids
    num_liquids: int = 4
    
    # Growth
    initial_budget: float = 500.0
    
    # Learning
    learning_rate: float = 0.01
    hebbian_rate: float = 0.05


class SmartBrain:
    """
    Complete Smart AGI Brain.
    
    Integrates all components into a unified system that can:
    - Process observations and produce actions
    - Self-organize through Hebbian learning
    - Grow and prune based on task demands
    - Use tools as first-class actions
    - Operate in dual roles (Experiencer/Saboteur)
    - Learn forward and inverse models simultaneously
    """
    
    def __init__(self, config: SmartBrainConfig):
        """Initialize the brain."""
        self.config = config
        
        # === Core Neural Components ===
        
        # Input encoder
        self.input_encoder = NeuronLayer(
            input_dim=config.observation_dim,
            output_dim=config.embedding_dim,
            sparsity=0.3,
            layer_id=0
        )
        
        # Sheet (GyrusNet) with microcolumns
        sheet_config = SheetConfig(
            size=config.sheet_size,
            input_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim // 2,
            state_dim=config.hidden_dim // 2,
            num_liquids=config.num_liquids
        )
        self.sheet = Sheet(sheet_config)
        
        # Router for column selection
        router_config = RouterConfig(
            num_columns=config.sheet_size ** 2,
            embedding_dim=config.embedding_dim,
            top_k=config.top_k
        )
        self.router = Router(router_config)
        
        # Set wire costs based on positions
        positions = [col.position for col in self.sheet.columns]
        center = (config.sheet_size // 2, config.sheet_size // 2)
        self.router.set_wire_costs(positions, center)
        
        # Output decoder
        self.output_decoder = NeuronLayer(
            input_dim=config.hidden_dim // 2 * config.top_k,
            output_dim=config.action_dim,
            sparsity=0.3,
            layer_id=1
        )
        
        # === Self-Organization Components ===
        
        # Liquid pool for skill clustering
        liquid_config = LiquidConfig(
            embedding_dim=config.embedding_dim,
            initial_num_liquids=config.num_liquids,
            max_liquids=config.num_liquids * 2
        )
        self.liquids = LiquidPool(liquid_config)
        
        # Growth controller
        growth_config = GrowthConfig(initial_budget=config.initial_budget)
        self.growth = GrowthController(growth_config)
        self.growth.set_network_size(
            num_columns=len(self.sheet.columns),
            neurons_per_column={i: config.hidden_dim for i in range(len(self.sheet.columns))},
            total_connections=len(self.sheet.columns) ** 2
        )
        
        # === Mirror Model (Forward/Inverse) ===
        mirror_config = MirrorConfig(
            state_dim=config.observation_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim // 2
        )
        self.mirror = UnifiedMirrorModel(mirror_config)
        
        # === Tool System ===
        self.tool_registry = ToolRegistry()
        self.tool_selector = ToolSelector(
            self.tool_registry,
            embedding_dim=config.embedding_dim
        )
        
        # === State ===
        self.current_role = Role.EXPERIENCER
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        
        # Buffers
        self.last_observation = None
        self.last_action = None
        self.last_embedding = None
        
    def set_role(self, role: Role):
        """Set operating role (Experiencer or Saboteur)."""
        self.current_role = role
    
    def forward(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process observation and produce action.
        
        Args:
            observation: Environment observation
        
        Returns:
            Tuple of (action_logits, info_dict)
        """
        self.step_count += 1
        self.last_observation = observation
        
        # Pad observation if needed
        if len(observation) < self.config.observation_dim:
            padded = np.zeros(self.config.observation_dim, dtype=np.float32)
            padded[:len(observation)] = observation
            observation = padded
        
        # Encode observation
        encoded = self.input_encoder.forward(observation)
        encoded_float = encoded.astype(np.float32)
        self.last_embedding = encoded_float
        
        # Route to columns
        query = encoded_float / (np.linalg.norm(encoded_float) + 1e-8)
        selected_indices, weights = self.router.select(query)
        
        # Forward through selected columns
        mode = ColumnMode(self.current_role.value)
        outputs, activations = self.sheet.forward(
            encoded_float,
            selected_indices=selected_indices,
            mode=mode
        )
        
        # Aggregate outputs
        weighted_outputs = outputs * weights[:, None]
        aggregated = weighted_outputs.sum(axis=0)
        
        # Decode to action logits
        flat_outputs = outputs.flatten()
        output_dim = self.config.hidden_dim // 2 * self.config.top_k
        if len(flat_outputs) < output_dim:
            padded = np.zeros(output_dim, dtype=np.float32)
            padded[:len(flat_outputs)] = flat_outputs
            flat_outputs = padded
        
        action_logits = self.output_decoder.forward(flat_outputs[:output_dim])
        action_logits = action_logits.astype(np.float32)
        
        # Check for tool use
        tool_name, tool_conf = self.tool_selector.select(query)
        tool_result = None
        if tool_name and tool_conf > 0.6:
            tool_inputs = self._construct_tool_inputs(tool_name, observation)
            tool_result = self.tool_selector.execute(tool_name, tool_inputs)
        
        info = {
            'selected_columns': selected_indices,
            'weights': weights,
            'activations': activations,
            'tool_used': tool_name,
            'tool_result': tool_result,
            'role': self.current_role.name,
            'aggregated': aggregated
        }
        
        return action_logits, info
    
    def _construct_tool_inputs(self, tool_name: str, obs: np.ndarray) -> Dict[str, Any]:
        """Construct tool inputs from observation."""
        if tool_name == 'math':
            return {'operation': 'add', 'operands': [float(obs[0]), float(obs[1])]}
        elif tool_name == 'memory':
            return {'action': 'read', 'key': 'state'}
        elif tool_name == 'reasoning':
            return {'operation': 'compare', 'inputs': [float(obs[0]), float(obs[1])]}
        return {}
    
    def get_action(self, action_logits: np.ndarray) -> int:
        """Convert action logits to discrete action."""
        return int(np.argmax(action_logits[:self.config.action_dim]))
    
    def learn(self, observation: np.ndarray, action: int, 
              next_observation: np.ndarray, reward: float, done: bool):
        """
        Learn from experience.
        
        Updates:
        - Mirror model (forward/inverse)
        - Hebbian connections
        - Growth controller metrics
        - Liquid activations
        """
        self.total_reward += reward
        
        # Convert action to vector
        action_vec = np.zeros(self.config.action_dim, dtype=np.float32)
        action_vec[action] = 1.0
        
        # Train mirror model
        if self.last_observation is not None:
            obs = self.last_observation
            if len(obs) < self.config.observation_dim:
                obs = np.zeros(self.config.observation_dim, dtype=np.float32)
                obs[:len(self.last_observation)] = self.last_observation
            
            next_obs = next_observation
            if len(next_obs) < self.config.observation_dim:
                next_obs_padded = np.zeros(self.config.observation_dim, dtype=np.float32)
                next_obs_padded[:len(next_observation)] = next_observation
                next_obs = next_obs_padded
            
            self.mirror.train_step(obs, action_vec, next_obs, self.config.learning_rate)
        
        # Record metrics for growth
        self.growth.metrics.record_task_result(success=reward > 0, reward=reward)
        utilization = np.mean(self.sheet.last_activations != 0) if hasattr(self.sheet, 'last_activations') else 0.5
        self.growth.metrics.record_utilization(utilization)
        
        # Hebbian updates on positive reward
        if reward > 0:
            self.sheet.hebbian_lateral_update()
            if self.last_embedding is not None:
                self.sheet.hebbian_column_update(self.last_embedding)
        
        # Update liquids
        for col in self.sheet.columns:
            if col.activation_count > 0:
                liquid_idx = col.dominant_liquid % len(self.liquids.liquids)
                self.liquids.liquids[liquid_idx].record_activation(self.step_count, reward)
        
        # Periodic structure maintenance
        if self.step_count % 100 == 0:
            self._structure_tick()
        
        self.last_action = action
    
    def _structure_tick(self):
        """Periodic structure maintenance."""
        self.growth.tick()
        self.liquids.structure_tick()
        
        # Check growth
        growth_ops = self.growth.plan_growth()
        if growth_ops:
            self.growth.execute_growth(growth_ops)
        
        # Check pruning
        activation_counts = {i: col.activation_count for i, col in enumerate(self.sheet.columns)}
        prune_ops = self.growth.plan_pruning(activation_counts, {})
        if prune_ops:
            self.growth.execute_pruning(prune_ops)
    
    def reset(self):
        """Reset for new episode."""
        self.episode_count += 1
        self.sheet.reset_states()
        self.last_observation = None
        self.last_action = None
        self.last_embedding = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics."""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'total_reward': self.total_reward,
            'current_role': self.current_role.name,
            'sheet_modularity': self.sheet.compute_modularity(),
            'num_columns': len(self.sheet.columns),
            'router_teams': self.router.get_stats()['num_teams'],
            'num_liquids': self.liquids.num_liquids,
            'growth_budget': self.growth.budget,
            'mirror_stats': self.mirror.get_stats(),
            'tool_stats': self.tool_selector.get_stats()
        }
    
    def __repr__(self) -> str:
        return (f"SmartBrain(columns={len(self.sheet.columns)}, "
                f"liquids={self.liquids.num_liquids}, "
                f"steps={self.step_count}, "
                f"reward={self.total_reward:.1f})")


def train_smart_brain(num_episodes: int = 100, max_steps: int = 500,
                      verbose: bool = True) -> Tuple[SmartBrain, List[Dict]]:
    """
    Train the Smart Brain on Flappy Bird.
    
    Args:
        num_episodes: Number of episodes
        max_steps: Max steps per episode
        verbose: Print progress
    
    Returns:
        Tuple of (trained_brain, episode_results)
    """
    from smart_agi.envs.flappy_bird import FlappyBirdEnv
    
    # Create brain
    config = SmartBrainConfig(
        observation_dim=8,
        action_dim=2,
        sheet_size=4,
        hidden_dim=32,
        embedding_dim=32,
        top_k=4,
        num_liquids=4
    )
    brain = SmartBrain(config)
    
    # Create environment
    env = FlappyBirdEnv()
    
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        brain.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Get action
            action_logits, info = brain.forward(obs)
            action = brain.get_action(action_logits)
            
            # Step environment
            next_obs, reward, done, env_info = env.step(action)
            episode_reward += reward
            
            # Learn
            brain.learn(obs, action, next_obs, reward, done)
            
            obs = next_obs
            
            if done:
                break
        
        result = {
            'episode': episode,
            'score': env.score,
            'steps': env.steps,
            'reward': episode_reward,
            'modularity': brain.sheet.compute_modularity()
        }
        results.append(result)
        
        if verbose and episode % 10 == 0:
            print(f"Episode {episode}: score={env.score}, reward={episode_reward:.1f}, "
                  f"modularity={result['modularity']:.3f}")
    
    return brain, results


if __name__ == "__main__":
    print("=== Smart AGI Brain Training ===\n")
    
    brain, results = train_smart_brain(num_episodes=50, max_steps=500, verbose=True)
    
    print(f"\n=== Training Complete ===")
    print(f"Final Brain: {brain}")
    print(f"\nStatistics:")
    stats = brain.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nPerformance Summary:")
    print(f"  Avg Score: {np.mean([r['score'] for r in results]):.1f}")
    print(f"  Max Score: {max([r['score'] for r in results])}")
    print(f"  Avg Modularity: {np.mean([r['modularity'] for r in results]):.3f}")
