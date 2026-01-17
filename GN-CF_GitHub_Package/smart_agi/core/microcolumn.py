"""
Microcolumn Module

Microcolumns are the fundamental computational units in the GyrusNet sheet.
Each microcolumn is a small neural network that processes inputs and produces
outputs, with an affinity vector that determines its specialization.

Inspired by cortical microcolumns in the brain, these units:
- Process local information
- Have soft assignment to "Liquids" (skill clusters)
- Maintain local state across time steps
- Participate in Hebbian learning with neighbors
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import IntEnum

from .neuron import NeuronLayer, NeuronConfig, TernaryNeuron
from .ternary import quantize_to_quinary, quinary_activation


class ColumnMode(IntEnum):
    """Operating mode for the microcolumn (dual-role system)."""
    SABOTEUR = -1      # Generate challenges
    EVALUATE = 0       # Assess difficulty/quality
    EXPERIENCER = 1    # Solve tasks


@dataclass
class MicrocolumnConfig:
    """Configuration for a microcolumn."""
    input_dim: int = 64           # Dimension of input from workspace
    hidden_dim: int = 32          # Hidden layer dimension
    output_dim: int = 16          # Output contribution dimension
    state_dim: int = 16           # Local persistent state dimension
    num_liquids: int = 8          # Number of liquids (skill clusters)
    initial_sparsity: float = 0.5 # Weight sparsity
    position: Tuple[int, int] = (0, 0)  # Position on the sheet


class Microcolumn:
    """
    A microcolumn (expert unit) in the GyrusNet sheet.
    
    Architecture:
        Input (from workspace) → Hidden Layer → Output (contribution)
                                     ↓
                              Local State Update
    
    Each microcolumn has:
    - Ternary weights with quinary activations
    - Soft affinity assignment to K liquids
    - Local persistent state
    - Position on the 2D sheet
    """
    
    def __init__(self, config: MicrocolumnConfig, column_id: int = 0):
        """
        Initialize a microcolumn.
        
        Args:
            config: Microcolumn configuration
            column_id: Unique identifier
        """
        self.config = config
        self.column_id = column_id
        self.position = config.position
        
        # Neural layers (ternary weights, quinary activations)
        self.input_layer = NeuronLayer(
            input_dim=config.input_dim + config.state_dim + 1,  # +1 for mode
            output_dim=config.hidden_dim,
            sparsity=config.initial_sparsity,
            layer_id=0
        )
        
        self.output_layer = NeuronLayer(
            input_dim=config.hidden_dim,
            output_dim=config.output_dim,
            sparsity=config.initial_sparsity,
            layer_id=1
        )
        
        self.state_layer = NeuronLayer(
            input_dim=config.hidden_dim,
            output_dim=config.state_dim,
            sparsity=config.initial_sparsity,
            layer_id=2
        )
        
        # Affinity vector over liquids (soft assignment, sums to 1)
        self.affinity = self._init_affinity()
        
        # Local persistent state
        self.state = np.zeros(config.state_dim, dtype=np.float32)
        
        # Embedding for routing (learned representation of this column)
        self.embedding = np.random.randn(config.input_dim).astype(np.float32)
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8
        
        # Statistics
        self.activation_count = 0
        self.total_output_magnitude = 0.0
        self.mode_activations = {-1: 0, 0: 0, 1: 0}  # Per-mode counts
        
    def _init_affinity(self) -> np.ndarray:
        """Initialize affinity vector (simplex over liquids)."""
        # Start with uniform distribution + small noise
        K = self.config.num_liquids
        affinity = np.ones(K) / K
        noise = np.random.randn(K) * 0.1
        affinity = affinity + noise
        affinity = np.maximum(affinity, 0)  # Non-negative
        affinity = affinity / (affinity.sum() + 1e-8)  # Normalize
        return affinity.astype(np.float32)
    
    def forward(self, workspace_input: np.ndarray, 
                mode: int = ColumnMode.EXPERIENCER) -> np.ndarray:
        """
        Process input and produce output contribution.
        
        Args:
            workspace_input: Input from the workspace (input_dim,)
            mode: Operating mode (-1=saboteur, 0=evaluate, +1=experiencer)
        
        Returns:
            Output contribution (output_dim,)
        """
        # Concatenate input with state and mode
        mode_signal = np.array([mode], dtype=np.float32)
        combined_input = np.concatenate([
            workspace_input, 
            self.state, 
            mode_signal
        ])
        
        # Forward through layers
        hidden = self.input_layer.forward(combined_input)
        hidden_float = hidden.astype(np.float32)
        
        output = self.output_layer.forward(hidden_float)
        new_state = self.state_layer.forward(hidden_float)
        
        # Update local state (exponential moving average)
        alpha = 0.3
        self.state = alpha * new_state.astype(np.float32) + (1 - alpha) * self.state
        
        # Update statistics
        self.activation_count += 1
        self.total_output_magnitude += np.abs(output).sum()
        self.mode_activations[mode] = self.mode_activations.get(mode, 0) + 1
        
        return output.astype(np.float32)
    
    def update_affinity(self, liquid_prototypes: np.ndarray, 
                        temperature: float = 1.0) -> np.ndarray:
        """
        Update affinity vector based on similarity to liquid prototypes.
        
        Args:
            liquid_prototypes: Matrix of liquid prototypes (K x embedding_dim)
            temperature: Softmax temperature (lower = sharper)
        
        Returns:
            Updated affinity vector
        """
        # Compute similarity to each liquid
        similarities = liquid_prototypes @ self.embedding
        
        # Softmax with temperature
        exp_sim = np.exp(similarities / temperature)
        self.affinity = exp_sim / (exp_sim.sum() + 1e-8)
        
        return self.affinity
    
    def hebbian_update(self, workspace_input: np.ndarray, 
                       learning_rate: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Apply Hebbian learning to all layers.
        
        Args:
            workspace_input: Input that produced current output
            learning_rate: Learning rate for updates
        
        Returns:
            Dictionary of updates per layer
        """
        mode_signal = np.array([1], dtype=np.float32)  # Default mode
        combined_input = np.concatenate([workspace_input, self.state, mode_signal])
        
        updates = {
            'input_layer': self.input_layer.hebbian_update(
                combined_input[None, :], learning_rate
            ),
            'output_layer': self.output_layer.hebbian_update(
                self.input_layer.activations, learning_rate
            ),
            'state_layer': self.state_layer.hebbian_update(
                self.input_layer.activations, learning_rate
            )
        }
        
        return updates
    
    def update_embedding(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """
        Update the column's embedding based on routing feedback.
        
        Args:
            gradient: Gradient from routing loss
            learning_rate: Learning rate
        """
        self.embedding -= learning_rate * gradient
        # Normalize
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8
    
    @property
    def dominant_liquid(self) -> int:
        """Index of the liquid with highest affinity."""
        return int(np.argmax(self.affinity))
    
    @property
    def affinity_entropy(self) -> float:
        """Entropy of affinity distribution (higher = more uncertain)."""
        # Avoid log(0)
        p = np.clip(self.affinity, 1e-10, 1.0)
        return -np.sum(p * np.log(p))
    
    @property
    def average_output_magnitude(self) -> float:
        """Average magnitude of outputs."""
        if self.activation_count == 0:
            return 0.0
        return self.total_output_magnitude / self.activation_count
    
    @property
    def sparsity(self) -> float:
        """Average weight sparsity across layers."""
        return (self.input_layer.sparsity + 
                self.output_layer.sparsity + 
                self.state_layer.sparsity) / 3
    
    def reset_state(self):
        """Reset local state to zeros."""
        self.state = np.zeros(self.config.state_dim, dtype=np.float32)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this microcolumn."""
        return {
            'column_id': self.column_id,
            'position': self.position,
            'activation_count': self.activation_count,
            'dominant_liquid': self.dominant_liquid,
            'affinity_entropy': self.affinity_entropy,
            'average_output_magnitude': self.average_output_magnitude,
            'sparsity': self.sparsity,
            'mode_activations': dict(self.mode_activations)
        }
    
    def clone(self) -> 'Microcolumn':
        """Create a copy of this microcolumn."""
        new_col = Microcolumn(self.config, self.column_id)
        new_col.input_layer.weights = self.input_layer.weights.copy()
        new_col.output_layer.weights = self.output_layer.weights.copy()
        new_col.state_layer.weights = self.state_layer.weights.copy()
        new_col.affinity = self.affinity.copy()
        new_col.state = self.state.copy()
        new_col.embedding = self.embedding.copy()
        return new_col
    
    def __repr__(self) -> str:
        return (f"Microcolumn(id={self.column_id}, "
                f"pos={self.position}, "
                f"dominant_liquid={self.dominant_liquid}, "
                f"activations={self.activation_count})")


class MicrocolumnFactory:
    """Factory for creating microcolumns with consistent configuration."""
    
    def __init__(self, base_config: MicrocolumnConfig):
        """
        Initialize factory.
        
        Args:
            base_config: Base configuration for all microcolumns
        """
        self.base_config = base_config
        self.next_id = 0
    
    def create(self, position: Tuple[int, int]) -> Microcolumn:
        """
        Create a new microcolumn at the specified position.
        
        Args:
            position: (x, y) position on the sheet
        
        Returns:
            New Microcolumn instance
        """
        config = MicrocolumnConfig(
            input_dim=self.base_config.input_dim,
            hidden_dim=self.base_config.hidden_dim,
            output_dim=self.base_config.output_dim,
            state_dim=self.base_config.state_dim,
            num_liquids=self.base_config.num_liquids,
            initial_sparsity=self.base_config.initial_sparsity,
            position=position
        )
        
        column = Microcolumn(config, self.next_id)
        self.next_id += 1
        return column
    
    def create_grid(self, size: int) -> List[Microcolumn]:
        """
        Create a grid of microcolumns.
        
        Args:
            size: Grid size (size x size)
        
        Returns:
            List of microcolumns in row-major order
        """
        columns = []
        for y in range(size):
            for x in range(size):
                columns.append(self.create((x, y)))
        return columns


if __name__ == "__main__":
    print("=== Microcolumn Tests ===\n")
    
    # Create configuration
    config = MicrocolumnConfig(
        input_dim=32,
        hidden_dim=16,
        output_dim=8,
        state_dim=8,
        num_liquids=4,
        initial_sparsity=0.5
    )
    
    # Create microcolumn
    col = Microcolumn(config, column_id=0)
    print(f"Created: {col}")
    print(f"Initial affinity: {col.affinity}")
    print(f"Affinity entropy: {col.affinity_entropy:.3f}\n")
    
    # Test forward pass in different modes
    x = np.random.randn(32).astype(np.float32)
    
    for mode in [ColumnMode.SABOTEUR, ColumnMode.EVALUATE, ColumnMode.EXPERIENCER]:
        output = col.forward(x, mode=mode)
        print(f"Mode {mode.name}: output shape={output.shape}, "
              f"mean={output.mean():.3f}, std={output.std():.3f}")
    
    print(f"\nAfter forwards: {col}")
    print(f"Stats: {col.get_stats()}\n")
    
    # Test affinity update
    liquid_prototypes = np.random.randn(4, 32).astype(np.float32)
    liquid_prototypes /= np.linalg.norm(liquid_prototypes, axis=1, keepdims=True)
    
    new_affinity = col.update_affinity(liquid_prototypes, temperature=0.5)
    print(f"Updated affinity: {new_affinity}")
    print(f"New dominant liquid: {col.dominant_liquid}\n")
    
    # Test Hebbian update
    updates = col.hebbian_update(x, learning_rate=0.1)
    print(f"Hebbian updates applied to {len(updates)} layers")
    
    # Test factory
    print("\n=== Factory Tests ===\n")
    factory = MicrocolumnFactory(config)
    grid = factory.create_grid(3)
    print(f"Created {len(grid)} microcolumns in 3x3 grid")
    for col in grid:
        print(f"  {col}")
