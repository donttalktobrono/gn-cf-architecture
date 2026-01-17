"""
Smart AGI Brain - Integrated Neural Architecture

This module integrates all components into a unified brain:
- Balanced ternary neurons with quinary activations
- Self-organizing Sheet (GyrusNet) with LE/LI connectivity
- Dynamic Liquids for skill clustering
- Hebbian coupling and team formation
- Dynamic growth/pruning with resource costs
- Tool use as first-class actions
- Dual-role (Experiencer/Saboteur) operation

The brain processes inputs, routes to appropriate columns,
aggregates outputs, and learns through Hebbian updates.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Import all components
from smart_agi.core.ternary import quantize_to_quinary, TernaryArray
from smart_agi.core.neuron import NeuronLayer
from smart_agi.core.microcolumn import Microcolumn, MicrocolumnConfig, ColumnMode
from smart_agi.core.sheet import Sheet, SheetConfig
from smart_agi.core.router import Router, RouterConfig
from smart_agi.core.growth import GrowthController, GrowthConfig
from smart_agi.liquids.liquid import LiquidPool, LiquidConfig
from smart_agi.tools.tool_system import ToolRegistry, ToolSelector
from smart_agi.training.adversarial import (
    AdversarialCurriculum, AdversarialConfig,
    FlappyBirdChallengeGenerator, Role
)


@dataclass
class BrainConfig:
    """Configuration for the integrated brain."""
    # Dimensions
    input_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 16
    embedding_dim: int = 64
    
    # Sheet configuration
    sheet_size: int = 8
    
    # Router configuration
    top_k: int = 8
    
    # Liquid configuration
    num_liquids: int = 8
    
    # Growth configuration
    initial_budget: float = 1000.0
    
    # Training configuration
    learning_rate: float = 0.01
    hebbian_rate: float = 0.05


class Workspace:
    """
    Central workspace that holds the current state.
    
    The workspace is a shared memory space that:
    - Receives inputs from the environment
    - Holds intermediate computations
    - Provides context to all columns
    - Stores outputs for action selection
    """
    
    def __init__(self, dim: int = 64):
        """
        Initialize workspace.
        
        Args:
            dim: Dimension of workspace vectors
        """
        self.dim = dim
        
        # State vectors
        self.input_buffer = np.zeros(dim, dtype=np.float32)
        self.context_buffer = np.zeros(dim, dtype=np.float32)
        self.output_buffer = np.zeros(dim, dtype=np.float32)
        
        # History for temporal context
        self.history: List[np.ndarray] = []
        self.history_size = 10
        
    def set_input(self, observation: np.ndarray):
        """Set input from environment observation."""
        # Pad or truncate to workspace dimension
        if len(observation) < self.dim:
            self.input_buffer[:len(observation)] = observation
            self.input_buffer[len(observation):] = 0
        else:
            self.input_buffer = observation[:self.dim].astype(np.float32)
    
    def update_context(self, new_context: np.ndarray, alpha: float = 0.3):
        """Update context with exponential moving average."""
        # Pad or truncate to workspace dimension
        if len(new_context) < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:len(new_context)] = new_context
            new_context = padded
        else:
            new_context = new_context[:self.dim]
        
        self.context_buffer = alpha * new_context + (1 - alpha) * self.context_buffer
        
        # Store in history
        self.history.append(self.context_buffer.copy())
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def set_output(self, output: np.ndarray):
        """Set output buffer."""
        if len(output) < self.dim:
            self.output_buffer[:len(output)] = output
        else:
            self.output_buffer = output[:self.dim].astype(np.float32)
    
    def get_combined_state(self) -> np.ndarray:
        """Get combined state for routing."""
        return np.concatenate([
            self.input_buffer,
            self.context_buffer
        ])
    
    def get_query(self) -> np.ndarray:
        """Get query vector for routing."""
        # Combine input and context
        combined = self.input_buffer + 0.5 * self.context_buffer
        return combined / (np.linalg.norm(combined) + 1e-8)
    
    def reset(self):
        """Reset workspace state."""
        self.input_buffer.fill(0)
        self.context_buffer.fill(0)
        self.output_buffer.fill(0)
        self.history.clear()


class Brain:
    """
    Integrated AGI Brain.
    
    Combines all components into a unified system that can:
    - Process observations and produce actions
    - Self-organize through Hebbian learning
    - Grow and prune based on task demands
    - Use tools as first-class actions
    - Operate in dual roles (experiencer/saboteur)
    """
    
    def __init__(self, config: BrainConfig):
        """
        Initialize the brain.
        
        Args:
            config: Brain configuration
        """
        self.config = config
        
        # Create workspace
        self.workspace = Workspace(config.embedding_dim)
        
        # Create input projection (to embedding space)
        self.input_projection = NeuronLayer(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            sparsity=0.3,
            layer_id=0
        )
        
        # Create sheet (GyrusNet)
        sheet_config = SheetConfig(
            size=config.sheet_size,
            input_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            state_dim=config.output_dim,
            num_liquids=config.num_liquids
        )
        self.sheet = Sheet(sheet_config)
        
        # Create router
        router_config = RouterConfig(
            num_columns=config.sheet_size ** 2,
            embedding_dim=config.embedding_dim,
            top_k=config.top_k
        )
        self.router = Router(router_config)
        
        # Set wire costs based on sheet positions
        positions = [col.position for col in self.sheet.columns]
        center = (config.sheet_size // 2, config.sheet_size // 2)
        self.router.set_wire_costs(positions, center)
        
        # Create liquid pool
        liquid_config = LiquidConfig(
            embedding_dim=config.embedding_dim,
            initial_num_liquids=config.num_liquids,
            max_liquids=config.num_liquids * 2
        )
        self.liquids = LiquidPool(liquid_config)
        
        # Create growth controller
        growth_config = GrowthConfig(
            initial_budget=config.initial_budget
        )
        self.growth = GrowthController(growth_config)
        self.growth.set_network_size(
            num_columns=len(self.sheet.columns),
            neurons_per_column={i: config.hidden_dim for i in range(len(self.sheet.columns))},
            total_connections=len(self.sheet.columns) ** 2
        )
        
        # Create tool system
        self.tool_registry = ToolRegistry()
        self.tool_selector = ToolSelector(
            self.tool_registry,
            embedding_dim=config.embedding_dim
        )
        
        # Create output projection (to action space)
        self.output_projection = NeuronLayer(
            input_dim=config.output_dim * config.top_k,
            output_dim=config.output_dim,
            sparsity=0.3,
            layer_id=1
        )
        
        # Current role
        self.current_role = Role.EXPERIENCER
        
        # Statistics
        self.step_count = 0
        self.total_reward = 0.0
        
    def set_role(self, role: Role):
        """Set the current operating role."""
        self.current_role = role
    
    def forward(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process observation and produce output.
        
        Args:
            observation: Environment observation
        
        Returns:
            Tuple of (output, info_dict)
        """
        self.step_count += 1
        
        # Project input to embedding space
        if len(observation) < self.config.input_dim:
            padded = np.zeros(self.config.input_dim, dtype=np.float32)
            padded[:len(observation)] = observation
            observation = padded
        
        embedded = self.input_projection.forward(observation[:self.config.input_dim])
        embedded_float = embedded.astype(np.float32)
        
        # Update workspace
        self.workspace.set_input(embedded_float)
        
        # Get routing query
        query = self.workspace.get_query()
        
        # Route to columns
        selected_indices, weights = self.router.select(query)
        
        # Forward through selected columns
        mode = ColumnMode(self.current_role.value)
        outputs, activations = self.sheet.forward(
            embedded_float,
            selected_indices=selected_indices,
            mode=mode
        )
        
        # Aggregate outputs
        weighted_outputs = outputs * weights[:, None]
        aggregated = weighted_outputs.sum(axis=0)
        
        # Update workspace context
        self.workspace.update_context(aggregated)
        
        # Check for tool use
        tool_name, tool_confidence = self.tool_selector.select(query)
        tool_result = None
        
        if tool_name and tool_confidence > 0.6:
            # Construct tool inputs from workspace state
            tool_inputs = self._construct_tool_inputs(tool_name)
            tool_result = self.tool_selector.execute(tool_name, tool_inputs)
        
        # Project to output space
        flat_outputs = outputs.flatten()
        if len(flat_outputs) < self.config.output_dim * self.config.top_k:
            padded = np.zeros(self.config.output_dim * self.config.top_k, dtype=np.float32)
            padded[:len(flat_outputs)] = flat_outputs
            flat_outputs = padded
        
        final_output = self.output_projection.forward(
            flat_outputs[:self.config.output_dim * self.config.top_k]
        )
        
        # Update workspace output
        self.workspace.set_output(final_output.astype(np.float32))
        
        # Collect info
        info = {
            'selected_columns': selected_indices,
            'weights': weights,
            'activations': activations,
            'tool_used': tool_name,
            'tool_result': tool_result,
            'role': self.current_role.name
        }
        
        return final_output.astype(np.float32), info
    
    def _construct_tool_inputs(self, tool_name: str) -> Dict[str, Any]:
        """Construct tool inputs from workspace state."""
        # Simple heuristic construction
        if tool_name == 'math':
            return {
                'operation': 'add',
                'operands': [float(self.workspace.input_buffer[0]), 
                            float(self.workspace.input_buffer[1])]
            }
        elif tool_name == 'memory':
            return {
                'action': 'read',
                'key': 'state'
            }
        elif tool_name == 'reasoning':
            return {
                'operation': 'compare',
                'inputs': [float(self.workspace.input_buffer[0]),
                          float(self.workspace.input_buffer[1])]
            }
        return {}
    
    def get_action(self, output: np.ndarray) -> int:
        """
        Convert output to discrete action.
        
        Args:
            output: Network output
        
        Returns:
            Discrete action index
        """
        # Simple argmax for discrete actions
        return int(np.argmax(output[:2]))  # Binary action for Flappy Bird
    
    def learn(self, reward: float, done: bool):
        """
        Learn from experience.
        
        Args:
            reward: Reward received
            done: Whether episode is done
        """
        self.total_reward += reward
        
        # Record metrics for growth controller
        self.growth.metrics.record_task_result(
            success=reward > 0,
            reward=reward
        )
        
        # Compute utilization
        utilization = np.mean(self.sheet.last_activations != 0)
        self.growth.metrics.record_utilization(utilization)
        
        # Hebbian updates
        if reward > 0:
            # Strengthen connections that led to reward
            self.sheet.hebbian_lateral_update()
            self.sheet.hebbian_column_update(self.workspace.input_buffer)
        
        # Update liquids
        for col in self.sheet.columns:
            if col.activation_count > 0:
                self.liquids.liquids[col.dominant_liquid % len(self.liquids.liquids)].record_activation(
                    self.step_count, reward
                )
        
        # Growth/pruning check
        if self.step_count % 100 == 0:
            self._structure_tick()
    
    def _structure_tick(self):
        """Perform periodic structure maintenance."""
        self.growth.tick()
        self.liquids.structure_tick()
        
        # Check growth
        growth_ops = self.growth.plan_growth()
        if growth_ops:
            self.growth.execute_growth(growth_ops)
        
        # Check pruning
        activation_counts = {
            i: col.activation_count
            for i, col in enumerate(self.sheet.columns)
        }
        prune_ops = self.growth.plan_pruning(activation_counts, {})
        if prune_ops:
            self.growth.execute_pruning(prune_ops)
    
    def reset(self):
        """Reset brain state for new episode."""
        self.workspace.reset()
        self.sheet.reset_states()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        return {
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'current_role': self.current_role.name,
            'sheet_modularity': self.sheet.compute_modularity(),
            'router_stats': self.router.get_stats(),
            'growth_stats': self.growth.get_stats(),
            'liquid_stats': self.liquids.get_stats(),
            'tool_stats': self.tool_selector.get_stats()
        }
    
    def __repr__(self) -> str:
        return (f"Brain(columns={len(self.sheet.columns)}, "
                f"liquids={self.liquids.num_liquids}, "
                f"steps={self.step_count})")


def train_on_flappy_bird(brain: Brain, num_episodes: int = 100,
                         max_steps: int = 1000) -> List[Dict[str, Any]]:
    """
    Train the brain on Flappy Bird.
    
    Args:
        brain: Brain instance
        num_episodes: Number of episodes
        max_steps: Maximum steps per episode
    
    Returns:
        List of episode results
    """
    from smart_agi.envs.flappy_bird import FlappyBirdEnv
    
    env = FlappyBirdEnv()
    results = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        brain.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Forward pass
            output, info = brain.forward(obs)
            
            # Get action
            action = brain.get_action(output)
            
            # Step environment
            obs, reward, done, env_info = env.step(action)
            episode_reward += reward
            
            # Learn
            brain.learn(reward, done)
            
            if done:
                break
        
        result = {
            'episode': episode,
            'score': env.score,
            'steps': env.steps,
            'reward': episode_reward,
            'won': env.state.value == 'won'
        }
        results.append(result)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: score={env.score}, reward={episode_reward:.1f}")
    
    return results


if __name__ == "__main__":
    print("=== Brain Integration Tests ===\n")
    
    # Create brain
    config = BrainConfig(
        input_dim=8,
        hidden_dim=32,
        output_dim=8,
        embedding_dim=32,
        sheet_size=4,
        top_k=4,
        num_liquids=4
    )
    
    brain = Brain(config)
    print(f"Created: {brain}\n")
    
    # Test forward pass
    obs = np.random.randn(8).astype(np.float32)
    output, info = brain.forward(obs)
    print(f"Forward pass:")
    print(f"  Input shape: {obs.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Selected columns: {info['selected_columns']}")
    print(f"  Tool used: {info['tool_used']}\n")
    
    # Test action selection
    action = brain.get_action(output)
    print(f"Action: {action}\n")
    
    # Test learning
    brain.learn(reward=1.0, done=False)
    print(f"After learning: {brain}\n")
    
    # Test on Flappy Bird
    print("Training on Flappy Bird (20 episodes)...")
    results = train_on_flappy_bird(brain, num_episodes=20, max_steps=500)
    
    print(f"\nTraining Results:")
    print(f"  Avg Score: {np.mean([r['score'] for r in results]):.1f}")
    print(f"  Avg Reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Max Score: {max([r['score'] for r in results])}")
    
    print(f"\nFinal Brain Stats:")
    stats = brain.get_stats()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
