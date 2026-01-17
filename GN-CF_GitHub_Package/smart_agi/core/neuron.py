"""
Ternary Neuron Module

This module implements individual neurons with ternary weights and quinary activations.
Neurons are the fundamental computational units that form microcolumns.

Key features:
- Ternary weights {-1, 0, +1} for efficient computation
- Quinary activations {-2, -1, 0, +1, +2} for expressiveness
- Hebbian learning with ternary updates
- Sparse connectivity through natural zeros
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .ternary import (
    quantize_to_ternary, 
    quantize_to_quinary,
    ternary_dot_product_vectorized,
    TernaryArray,
    quinary_activation
)


@dataclass
class NeuronConfig:
    """Configuration for a ternary neuron."""
    input_dim: int
    initial_sparsity: float = 0.5  # Fraction of zero weights
    bias_range: Tuple[int, int] = (-2, 2)  # Quinary bias range
    learning_rate: float = 0.1
    hebbian_threshold: float = 2.0  # Correlation threshold for updates
    

class TernaryNeuron:
    """
    A single neuron with ternary weights and quinary activation.
    
    The neuron computes:
        z = sum(w_i * x_i) + b
        a = quinary_activate(z)
    
    where w_i ∈ {-1, 0, +1} and a ∈ {-2, -1, 0, +1, +2}
    
    Attributes:
        weights: Ternary weight vector
        bias: Quinary bias value
        activation: Current activation value
        activation_count: Number of times neuron has been activated
        pre_activations: Buffer of recent pre-synaptic activations (for Hebbian)
    """
    
    def __init__(self, config: NeuronConfig, neuron_id: int = 0):
        """
        Initialize a ternary neuron.
        
        Args:
            config: Neuron configuration
            neuron_id: Unique identifier for this neuron
        """
        self.config = config
        self.neuron_id = neuron_id
        
        # Initialize ternary weights with specified sparsity
        self.weights = self._init_weights()
        
        # Initialize quinary bias
        self.bias = np.random.randint(
            config.bias_range[0], 
            config.bias_range[1] + 1
        )
        
        # State tracking
        self.activation = 0
        self.activation_count = 0
        self.last_input = None
        
        # Hebbian learning buffers
        self.pre_activation_history = []
        self.post_activation_history = []
        self.history_size = 10
        
    def _init_weights(self) -> np.ndarray:
        """Initialize ternary weights with specified sparsity."""
        sparsity = self.config.initial_sparsity
        weights = np.random.choice(
            [-1, 0, 1],
            size=self.config.input_dim,
            p=[(1-sparsity)/2, sparsity, (1-sparsity)/2]
        )
        return weights.astype(np.int8)
    
    def forward(self, inputs: np.ndarray) -> int:
        """
        Compute neuron output.
        
        Args:
            inputs: Input vector (can be continuous or quinary)
        
        Returns:
            Quinary activation value {-2, -1, 0, +1, +2}
        """
        # Store input for Hebbian learning
        self.last_input = inputs
        
        # Compute weighted sum (efficient with ternary weights)
        z = ternary_dot_product_vectorized(self.weights, inputs)
        
        # Add bias
        z += self.bias
        
        # Apply quinary activation
        self.activation = quantize_to_quinary(z)
        
        # Track activation
        if self.activation != 0:
            self.activation_count += 1
        
        # Update history for Hebbian learning
        self._update_history(inputs, self.activation)
        
        return self.activation
    
    def _update_history(self, pre: np.ndarray, post: int):
        """Update activation history for Hebbian learning."""
        self.pre_activation_history.append(pre.copy())
        self.post_activation_history.append(post)
        
        # Keep history bounded
        if len(self.pre_activation_history) > self.history_size:
            self.pre_activation_history.pop(0)
            self.post_activation_history.pop(0)
    
    def hebbian_update(self) -> np.ndarray:
        """
        Apply Hebbian learning rule to update weights.
        
        The rule: "Cells that fire together wire together"
        
        For ternary weights:
        - If pre and post are both strongly active (same sign): strengthen (+1)
        - If pre and post have opposite strong activations: weaken (-1)
        - Otherwise: no change (0)
        
        Returns:
            Array of weight changes applied
        """
        if len(self.pre_activation_history) == 0:
            return np.zeros_like(self.weights)
        
        # Compute average correlation over history
        weight_deltas = np.zeros_like(self.weights, dtype=np.float32)
        
        for pre, post in zip(self.pre_activation_history, 
                            self.post_activation_history):
            # Correlation: pre * post
            correlation = pre * post
            weight_deltas += correlation
        
        weight_deltas /= len(self.pre_activation_history)
        
        # Quantize deltas to ternary updates
        threshold = self.config.hebbian_threshold
        updates = np.where(weight_deltas > threshold, 1,
                          np.where(weight_deltas < -threshold, -1, 0))
        
        # Apply stochastic updates based on learning rate
        mask = np.random.random(self.weights.shape) < self.config.learning_rate
        updates = updates * mask
        
        # Update weights (clamp to ternary range)
        self.weights = np.clip(self.weights + updates, -1, 1).astype(np.int8)
        
        return updates.astype(np.int8)
    
    @property
    def sparsity(self) -> float:
        """Fraction of zero weights."""
        return np.mean(self.weights == 0)
    
    @property
    def excitatory_fraction(self) -> float:
        """Fraction of positive weights."""
        return np.mean(self.weights == 1)
    
    @property
    def inhibitory_fraction(self) -> float:
        """Fraction of negative weights."""
        return np.mean(self.weights == -1)
    
    def prune_weak_connections(self, min_activation_correlation: float = 0.1):
        """
        Prune connections that rarely contribute to activation.
        
        Sets weights to zero if they don't correlate with neuron activation.
        
        Args:
            min_activation_correlation: Minimum correlation to keep connection
        """
        if len(self.pre_activation_history) < 5:
            return
        
        # Compute correlation between each input and post-activation
        pre_stack = np.stack(self.pre_activation_history)
        post_array = np.array(self.post_activation_history)
        
        correlations = np.abs(np.mean(pre_stack * post_array[:, None], axis=0))
        
        # Zero out weights with low correlation
        mask = correlations < min_activation_correlation
        self.weights[mask] = 0
    
    def clone(self) -> 'TernaryNeuron':
        """Create a copy of this neuron."""
        new_neuron = TernaryNeuron(self.config, self.neuron_id)
        new_neuron.weights = self.weights.copy()
        new_neuron.bias = self.bias
        new_neuron.activation_count = self.activation_count
        return new_neuron
    
    def __repr__(self) -> str:
        return (f"TernaryNeuron(id={self.neuron_id}, "
                f"input_dim={self.config.input_dim}, "
                f"sparsity={self.sparsity:.2%}, "
                f"activations={self.activation_count})")


class NeuronLayer:
    """
    A layer of ternary neurons.
    
    Provides efficient batch computation over multiple neurons.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 sparsity: float = 0.5, layer_id: int = 0):
        """
        Initialize a layer of ternary neurons.
        
        Args:
            input_dim: Dimension of input
            output_dim: Number of neurons in layer
            sparsity: Target weight sparsity
            layer_id: Identifier for this layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_id = layer_id
        
        # Initialize weight matrix (output_dim x input_dim)
        self.weights = self._init_weights(sparsity)
        
        # Initialize biases (quinary)
        self.biases = np.random.randint(-2, 3, size=output_dim).astype(np.int8)
        
        # Activation tracking
        self.activations = np.zeros(output_dim, dtype=np.int8)
        self.activation_counts = np.zeros(output_dim, dtype=np.int64)
        
    def _init_weights(self, sparsity: float) -> np.ndarray:
        """Initialize ternary weight matrix."""
        weights = np.random.choice(
            [-1, 0, 1],
            size=(self.output_dim, self.input_dim),
            p=[(1-sparsity)/2, sparsity, (1-sparsity)/2]
        )
        return weights.astype(np.int8)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute layer output.
        
        Args:
            inputs: Input vector (input_dim,) or batch (batch, input_dim)
        
        Returns:
            Quinary activations (output_dim,) or (batch, output_dim)
        """
        # Handle batch dimension
        if inputs.ndim == 1:
            inputs = inputs[None, :]
            squeeze = True
        else:
            squeeze = False
        
        # Matrix multiplication (efficient with ternary weights)
        z = inputs @ self.weights.T  # (batch, output_dim)
        
        # Add biases
        z += self.biases
        
        # Apply quinary activation
        self.activations = quinary_activation(z)
        
        # Track activations
        self.activation_counts += np.sum(self.activations != 0, axis=0)
        
        if squeeze:
            return self.activations[0]
        return self.activations
    
    def hebbian_update(self, inputs: np.ndarray, learning_rate: float = 0.1,
                       threshold: float = 2.0) -> np.ndarray:
        """
        Apply Hebbian learning to the weight matrix.
        
        Args:
            inputs: Input that produced current activations
            learning_rate: Probability of applying update
            threshold: Correlation threshold for updates
        
        Returns:
            Matrix of weight updates applied
        """
        # Ensure inputs is 2D
        if inputs.ndim == 1:
            inputs = inputs[None, :]
        
        # Ensure activations is 2D
        activations = self.activations
        if activations.ndim == 1:
            activations = activations[None, :]
        
        # Compute correlation: outer product of post and pre activations
        # Shape: (batch, output_dim, input_dim)
        correlation = activations[:, :, None] * inputs[:, None, :]
        
        # Average over batch
        avg_correlation = np.mean(correlation, axis=0)
        
        # Quantize to ternary updates
        updates = np.where(avg_correlation > threshold, 1,
                          np.where(avg_correlation < -threshold, -1, 0))
        
        # Stochastic application
        mask = np.random.random(self.weights.shape) < learning_rate
        updates = updates * mask
        
        # Apply updates
        self.weights = np.clip(self.weights + updates, -1, 1).astype(np.int8)
        
        return updates.astype(np.int8)
    
    @property
    def sparsity(self) -> float:
        """Fraction of zero weights."""
        return np.mean(self.weights == 0)
    
    @property
    def dead_neurons(self) -> int:
        """Number of neurons that have never activated."""
        return np.sum(self.activation_counts == 0)
    
    def __repr__(self) -> str:
        return (f"NeuronLayer(id={self.layer_id}, "
                f"shape=({self.input_dim}, {self.output_dim}), "
                f"sparsity={self.sparsity:.2%}, "
                f"dead={self.dead_neurons})")


if __name__ == "__main__":
    print("=== Ternary Neuron Tests ===\n")
    
    # Test single neuron
    config = NeuronConfig(input_dim=10, initial_sparsity=0.3)
    neuron = TernaryNeuron(config, neuron_id=0)
    print(f"Created: {neuron}")
    print(f"Weights: {neuron.weights}")
    print(f"Bias: {neuron.bias}\n")
    
    # Test forward pass
    for i in range(5):
        x = np.random.randn(10)
        a = neuron.forward(x)
        print(f"Input {i}: activation = {a}")
    
    print(f"\nAfter 5 forwards: {neuron}\n")
    
    # Test Hebbian update
    updates = neuron.hebbian_update()
    print(f"Hebbian updates: {updates}")
    print(f"New weights: {neuron.weights}\n")
    
    print("=== Neuron Layer Tests ===\n")
    
    # Test layer
    layer = NeuronLayer(input_dim=20, output_dim=10, sparsity=0.5)
    print(f"Created: {layer}")
    
    # Batch forward
    batch = np.random.randn(5, 20)
    outputs = layer.forward(batch)
    print(f"Batch output shape: {outputs.shape}")
    print(f"Outputs:\n{outputs}\n")
    
    # Hebbian update
    updates = layer.hebbian_update(batch)
    print(f"Update sparsity: {np.mean(updates == 0):.2%}")
    print(f"After update: {layer}")
