"""
Mirror Generalization System - AI-Native Implementation v2

Key Insight: AI is created in human's image, but AI is not human.
We capture the PRINCIPLE of mirror generalization, not the biology.

The Principle: Forward and Inverse models share a unified representation.
- Forward: (state, action) → next_state
- Inverse: (state, next_state) → action

Both paths project through the SAME embedding space.
This creates natural consistency without forced weight symmetry.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MirrorMode(Enum):
    """Direction of inference."""
    FORWARD = 1      # Predict outcomes
    INVERSE = -1     # Plan actions


@dataclass
class MirrorConfig:
    """Configuration for mirror model."""
    state_dim: int = 8
    action_dim: int = 2
    hidden_dim: int = 32
    embedding_dim: int = 16
    learning_rate: float = 0.01


class SimpleMLP:
    """Simple multi-layer perceptron."""
    
    def __init__(self, dims: List[int], activation: str = 'tanh'):
        """
        Initialize MLP.
        
        Args:
            dims: Layer dimensions [input, hidden1, ..., output]
            activation: Activation function
        """
        self.dims = dims
        self.activation = activation
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            W = np.random.randn(dims[i], dims[i+1]).astype(np.float32) * scale
            b = np.zeros(dims[i+1], dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)
        
        # Cache for backprop
        self.activations = []
        self.pre_activations = []
        
    def _activate(self, x: np.ndarray, is_output: bool = False) -> np.ndarray:
        """Apply activation."""
        if is_output:
            return x  # Linear output
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        return x
    
    def _activate_deriv(self, activated: np.ndarray) -> np.ndarray:
        """Activation derivative."""
        if self.activation == 'tanh':
            return 1 - activated ** 2
        elif self.activation == 'relu':
            return (activated > 0).astype(np.float32)
        return np.ones_like(activated)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.activations = [x]
        self.pre_activations = []
        
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            pre = h @ W + b
            self.pre_activations.append(pre)
            h = self._activate(pre, is_output=(i == len(self.weights) - 1))
            self.activations.append(h)
        
        return h
    
    def backward(self, grad_output: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
        """Backward pass with weight updates."""
        grad = grad_output
        
        for i in reversed(range(len(self.weights))):
            # Gradient through activation (skip for output layer)
            if i < len(self.weights) - 1:
                grad = grad * self._activate_deriv(self.activations[i + 1])
            
            # Weight gradient
            grad_W = np.outer(self.activations[i], grad)
            grad_b = grad
            
            # Update weights
            self.weights[i] -= learning_rate * grad_W
            self.biases[i] -= learning_rate * grad_b
            
            # Gradient for previous layer
            grad = grad @ self.weights[i].T
        
        return grad


class UnifiedMirrorModel:
    """
    Unified Forward-Inverse Model.
    
    Architecture:
    - Shared encoder: maps (state, context) → embedding
    - Forward decoder: embedding → next_state
    - Inverse decoder: embedding → action
    
    The shared encoder creates the "mirror" - both directions
    project through the same representation.
    """
    
    def __init__(self, config: MirrorConfig):
        """Initialize model."""
        self.config = config
        
        # Shared encoder: (state + context) → embedding
        # Context is action (forward) or next_state (inverse)
        encoder_input = config.state_dim + max(config.action_dim, config.state_dim)
        self.encoder = SimpleMLP([encoder_input, config.hidden_dim, config.embedding_dim])
        
        # Forward decoder: embedding → next_state
        self.forward_decoder = SimpleMLP([config.embedding_dim, config.hidden_dim, config.state_dim])
        
        # Inverse decoder: embedding → action
        self.inverse_decoder = SimpleMLP([config.embedding_dim, config.hidden_dim, config.action_dim])
        
        # Statistics
        self.forward_count = 0
        self.inverse_count = 0
        self.losses = []
        
    def _prepare_input(self, state: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Prepare input by concatenating state and context."""
        # Pad context to max size
        max_context = max(self.config.action_dim, self.config.state_dim)
        padded_context = np.zeros(max_context, dtype=np.float32)
        padded_context[:len(context)] = context
        return np.concatenate([state, padded_context])
    
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Forward model: (state, action) → next_state."""
        self.forward_count += 1
        
        # Encode
        x = self._prepare_input(state, action)
        embedding = self.encoder.forward(x)
        
        # Decode to next state
        next_state = self.forward_decoder.forward(embedding)
        
        return next_state
    
    def compute_action(self, state: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """Inverse model: (state, next_state) → action."""
        self.inverse_count += 1
        
        # Encode
        x = self._prepare_input(state, next_state)
        embedding = self.encoder.forward(x)
        
        # Decode to action
        action = self.inverse_decoder.forward(embedding)
        
        return action
    
    def get_embedding(self, state: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Get the shared embedding for a state-context pair."""
        x = self._prepare_input(state, context)
        return self.encoder.forward(x)
    
    def train_step(self, state: np.ndarray, action: np.ndarray,
                   next_state: np.ndarray, learning_rate: float = 0.01) -> Dict[str, float]:
        """
        Train on a single transition.
        
        Trains both forward and inverse models through the shared encoder.
        """
        # === Forward model training ===
        pred_next = self.predict_next_state(state, action)
        forward_loss = np.mean((pred_next - next_state) ** 2)
        
        # Backward through forward decoder
        forward_grad = 2 * (pred_next - next_state) / len(next_state)
        embed_grad_forward = self.forward_decoder.backward(forward_grad, learning_rate)
        
        # Backward through encoder (forward path)
        x_forward = self._prepare_input(state, action)
        _ = self.encoder.forward(x_forward)  # Recompute activations
        self.encoder.backward(embed_grad_forward, learning_rate)
        
        # === Inverse model training ===
        pred_action = self.compute_action(state, next_state)
        inverse_loss = np.mean((pred_action - action) ** 2)
        
        # Backward through inverse decoder
        inverse_grad = 2 * (pred_action - action) / len(action)
        embed_grad_inverse = self.inverse_decoder.backward(inverse_grad, learning_rate)
        
        # Backward through encoder (inverse path)
        x_inverse = self._prepare_input(state, next_state)
        _ = self.encoder.forward(x_inverse)  # Recompute activations
        self.encoder.backward(embed_grad_inverse, learning_rate)
        
        losses = {
            'forward': float(forward_loss),
            'inverse': float(inverse_loss),
            'total': float(forward_loss + inverse_loss)
        }
        self.losses.append(losses)
        
        return losses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        recent = self.losses[-100:] if self.losses else []
        return {
            'forward_count': self.forward_count,
            'inverse_count': self.inverse_count,
            'train_steps': len(self.losses),
            'avg_forward_loss': np.mean([l['forward'] for l in recent]) if recent else 0,
            'avg_inverse_loss': np.mean([l['inverse'] for l in recent]) if recent else 0
        }


class DualRoleBrain:
    """
    Brain that operates in Experiencer or Saboteur mode.
    
    Both modes share the same underlying mirror model.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        """Initialize brain."""
        config = MirrorConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            embedding_dim=hidden_dim // 2
        )
        self.mirror = UnifiedMirrorModel(config)
        self.mode = MirrorMode.FORWARD
        
        # Reward tracking
        self.experiencer_reward = 0.0
        self.saboteur_reward = 0.0
        
    def set_mode(self, mode: MirrorMode):
        """Set operating mode."""
        self.mode = mode
    
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next state (forward model)."""
        return self.mirror.predict_next_state(state, action)
    
    def plan(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Plan action to reach goal (inverse model)."""
        return self.mirror.compute_action(state, goal)
    
    def act(self, state: np.ndarray, goal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Choose action based on current mode.
        
        Experiencer: plan action toward goal
        Saboteur: plan action away from expected path
        """
        if goal is None:
            goal = state  # Default: maintain current state
        
        if self.mode == MirrorMode.FORWARD:
            # Experiencer: go toward goal
            return self.plan(state, goal)
        else:
            # Saboteur: perturb the goal to create challenge
            perturbed_goal = goal + np.random.randn(len(goal)).astype(np.float32) * 0.2
            return self.plan(state, perturbed_goal)
    
    def learn(self, state: np.ndarray, action: np.ndarray,
              next_state: np.ndarray, reward: float, lr: float = 0.01) -> Dict[str, float]:
        """Learn from transition."""
        losses = self.mirror.train_step(state, action, next_state, lr)
        
        if self.mode == MirrorMode.FORWARD:
            self.experiencer_reward += reward
        else:
            self.saboteur_reward += reward
        
        return losses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        return {
            'mode': 'EXPERIENCER' if self.mode == MirrorMode.FORWARD else 'SABOTEUR',
            'experiencer_reward': self.experiencer_reward,
            'saboteur_reward': self.saboteur_reward,
            'mirror_stats': self.mirror.get_stats()
        }


if __name__ == "__main__":
    print("=== Mirror Generalization Tests ===\n")
    
    # Test UnifiedMirrorModel
    config = MirrorConfig(state_dim=4, action_dim=2, hidden_dim=16, embedding_dim=8)
    model = UnifiedMirrorModel(config)
    
    state = np.array([0.5, 0.3, 0.0, 0.1], dtype=np.float32)
    action = np.array([1.0, 0.0], dtype=np.float32)
    next_state = np.array([0.6, 0.2, 0.1, 0.15], dtype=np.float32)
    
    print("Before training:")
    pred = model.predict_next_state(state, action)
    comp = model.compute_action(state, next_state)
    print(f"  Predicted next: {pred}")
    print(f"  Computed action: {comp}")
    
    print("\nTraining for 1000 steps...")
    for i in range(1000):
        losses = model.train_step(state, action, next_state, learning_rate=0.02)
        if i % 200 == 0:
            print(f"  Step {i}: forward={losses['forward']:.4f}, inverse={losses['inverse']:.4f}")
    
    print("\nAfter training:")
    pred = model.predict_next_state(state, action)
    comp = model.compute_action(state, next_state)
    print(f"  Predicted next: {pred}")
    print(f"  Actual next:    {next_state}")
    print(f"  Computed action: {comp}")
    print(f"  Actual action:   {action}")
    
    # Test DualRoleBrain
    print("\n\nDualRoleBrain test:")
    brain = DualRoleBrain(state_dim=4, action_dim=2)
    
    brain.set_mode(MirrorMode.FORWARD)
    exp_action = brain.act(state, next_state)
    print(f"  Experiencer action: {exp_action}")
    
    brain.set_mode(MirrorMode.INVERSE)
    sab_action = brain.act(state, next_state)
    print(f"  Saboteur action: {sab_action}")
