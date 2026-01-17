"""
Affinity Liquids Module

Liquids are dynamic skill primitives that represent clusters of related capabilities.
Unlike fixed expert modules, Liquids can spawn, merge, and evolve based on task demands.

Key concepts:
- Each Liquid has a prototype vector (centroid of skill cluster)
- Microcolumns have soft affinity assignments to Liquids
- Liquids can spawn when novel skills are needed
- Liquids can merge when they become redundant
- Liquids can be pruned when unused

This creates a self-organizing system where skills naturally cluster and specialize.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import time


class LiquidState(Enum):
    """State of a liquid in its lifecycle."""
    NASCENT = "nascent"      # Newly spawned, still forming
    ACTIVE = "active"        # Stable and in use
    DORMANT = "dormant"      # Not recently used
    MERGING = "merging"      # Being merged into another
    PRUNING = "pruning"      # Being removed


@dataclass
class LiquidConfig:
    """Configuration for the Liquids system."""
    embedding_dim: int = 64          # Dimension of prototype vectors
    initial_num_liquids: int = 4     # Starting number of liquids
    max_liquids: int = 16            # Maximum number of liquids
    min_liquids: int = 2             # Minimum number of liquids
    
    # Lifecycle parameters
    spawn_threshold: float = 0.7     # Novelty threshold to spawn new liquid
    merge_threshold: float = 0.85    # Similarity threshold to merge liquids
    prune_threshold: int = 100       # Ticks without use before pruning
    nascent_duration: int = 50       # Ticks before nascent becomes active
    
    # Learning parameters
    prototype_learning_rate: float = 0.1
    affinity_temperature: float = 1.0
    
    # Permission levels (which tools each liquid can access)
    default_permissions: List[str] = field(default_factory=lambda: ['math', 'memory_read'])


@dataclass
class LiquidStats:
    """Statistics for a single liquid."""
    activation_count: int = 0
    total_reward: float = 0.0
    last_activation_tick: int = 0
    member_count: int = 0
    average_affinity: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)


class Liquid:
    """
    A single Liquid (skill primitive).
    
    Each Liquid represents a cluster of related capabilities and has:
    - A prototype vector (centroid of the skill cluster)
    - Permissions for which tools it can access
    - Budgets for resource usage
    - Statistics about its usage and performance
    """
    
    def __init__(self, liquid_id: int, config: LiquidConfig, 
                 prototype: Optional[np.ndarray] = None):
        """
        Initialize a Liquid.
        
        Args:
            liquid_id: Unique identifier
            config: Liquid configuration
            prototype: Initial prototype vector (random if None)
        """
        self.liquid_id = liquid_id
        self.config = config
        
        # Prototype vector (centroid of skill cluster)
        if prototype is not None:
            self.prototype = prototype.copy()
        else:
            self.prototype = np.random.randn(config.embedding_dim).astype(np.float32)
        self.prototype /= np.linalg.norm(self.prototype) + 1e-8
        
        # State
        self.state = LiquidState.NASCENT
        self.creation_tick = 0
        
        # Permissions and budgets
        self.permissions: Set[str] = set(config.default_permissions)
        self.budgets: Dict[str, int] = {
            'compute': 1000,
            'memory': 100,
            'tool_calls': 50
        }
        self.usage: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self.stats = LiquidStats()
        
        # Member tracking (column indices with high affinity)
        self.member_columns: Set[int] = set()
        
        # History for prototype updates
        self.embedding_history: List[np.ndarray] = []
        self.history_size = 50
        
    def update_prototype(self, embedding: np.ndarray, weight: float = 1.0):
        """
        Update prototype based on new embedding.
        
        Uses exponential moving average.
        
        Args:
            embedding: New embedding to incorporate
            weight: Weight for this update
        """
        lr = self.config.prototype_learning_rate * weight
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Update prototype
        self.prototype = (1 - lr) * self.prototype + lr * embedding
        self.prototype /= np.linalg.norm(self.prototype) + 1e-8
        
        # Store in history
        self.embedding_history.append(embedding.copy())
        if len(self.embedding_history) > self.history_size:
            self.embedding_history.pop(0)
    
    def compute_affinity(self, embedding: np.ndarray) -> float:
        """
        Compute affinity score for an embedding.
        
        Args:
            embedding: Embedding to compute affinity for
        
        Returns:
            Affinity score (higher = more similar)
        """
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(self.prototype, embedding)
        
        return float(similarity)
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this liquid can use a tool."""
        if tool_name not in self.permissions:
            return False
        
        budget_key = 'tool_calls'
        if self.usage[budget_key] >= self.budgets.get(budget_key, 0):
            return False
        
        return True
    
    def use_tool(self, tool_name: str) -> bool:
        """Record tool usage."""
        if not self.can_use_tool(tool_name):
            return False
        
        self.usage['tool_calls'] += 1
        self.stats.tool_usage[tool_name] = self.stats.tool_usage.get(tool_name, 0) + 1
        return True
    
    def record_activation(self, tick: int, reward: float = 0.0):
        """Record an activation of this liquid."""
        self.stats.activation_count += 1
        self.stats.total_reward += reward
        self.stats.last_activation_tick = tick
        
        # Transition from nascent to active
        if self.state == LiquidState.NASCENT:
            if tick - self.creation_tick >= self.config.nascent_duration:
                self.state = LiquidState.ACTIVE
    
    def check_dormancy(self, current_tick: int) -> bool:
        """Check if liquid should become dormant."""
        if self.state == LiquidState.ACTIVE:
            ticks_since_use = current_tick - self.stats.last_activation_tick
            if ticks_since_use > self.config.prune_threshold // 2:
                self.state = LiquidState.DORMANT
                return True
        return False
    
    def should_prune(self, current_tick: int) -> bool:
        """Check if liquid should be pruned."""
        if self.state in [LiquidState.NASCENT, LiquidState.MERGING, LiquidState.PRUNING]:
            return False
        
        ticks_since_use = current_tick - self.stats.last_activation_tick
        return ticks_since_use > self.config.prune_threshold
    
    def reset_budgets(self):
        """Reset usage budgets (called periodically)."""
        self.usage = defaultdict(int)
    
    @property
    def average_reward(self) -> float:
        """Average reward per activation."""
        if self.stats.activation_count == 0:
            return 0.0
        return self.stats.total_reward / self.stats.activation_count
    
    @property
    def prototype_stability(self) -> float:
        """Measure of how stable the prototype is."""
        if len(self.embedding_history) < 2:
            return 0.0
        
        # Compute variance of recent embeddings
        embeddings = np.stack(self.embedding_history)
        variance = np.var(embeddings, axis=0).mean()
        
        # Lower variance = more stable
        return 1.0 / (1.0 + variance)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this liquid."""
        return {
            'liquid_id': self.liquid_id,
            'state': self.state.value,
            'activation_count': self.stats.activation_count,
            'average_reward': self.average_reward,
            'member_count': len(self.member_columns),
            'prototype_stability': self.prototype_stability,
            'permissions': list(self.permissions),
            'tool_usage': dict(self.stats.tool_usage)
        }
    
    def __repr__(self) -> str:
        return (f"Liquid(id={self.liquid_id}, state={self.state.value}, "
                f"activations={self.stats.activation_count}, "
                f"members={len(self.member_columns)})")


class LiquidPool:
    """
    Pool of Liquids with lifecycle management.
    
    Handles:
    - Spawning new liquids when novel skills are needed
    - Merging similar liquids to reduce redundancy
    - Pruning unused liquids
    - Computing affinities for microcolumns
    """
    
    def __init__(self, config: LiquidConfig):
        """
        Initialize the liquid pool.
        
        Args:
            config: Configuration for liquids
        """
        self.config = config
        self.liquids: Dict[int, Liquid] = {}
        self.next_id = 0
        self.current_tick = 0
        
        # Novelty buffer for spawn decisions
        self.novelty_buffer: List[Tuple[np.ndarray, float]] = []
        self.novelty_buffer_size = 100
        
        # Event log
        self.events: List[Dict[str, Any]] = []
        
        # Initialize with starting liquids
        for _ in range(config.initial_num_liquids):
            self._spawn_liquid()
    
    def _spawn_liquid(self, prototype: Optional[np.ndarray] = None) -> Liquid:
        """
        Spawn a new liquid.
        
        Args:
            prototype: Initial prototype (random if None)
        
        Returns:
            Newly created Liquid
        """
        liquid = Liquid(self.next_id, self.config, prototype)
        liquid.creation_tick = self.current_tick
        self.liquids[self.next_id] = liquid
        
        self.events.append({
            'type': 'spawn',
            'liquid_id': self.next_id,
            'tick': self.current_tick
        })
        
        self.next_id += 1
        return liquid
    
    def compute_affinities(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute affinity scores for all liquids.
        
        Args:
            embedding: Embedding to compute affinities for
        
        Returns:
            Array of affinity scores (one per liquid)
        """
        affinities = []
        for liquid_id in sorted(self.liquids.keys()):
            liquid = self.liquids[liquid_id]
            affinity = liquid.compute_affinity(embedding)
            affinities.append(affinity)
        
        # Apply softmax with temperature
        affinities = np.array(affinities)
        temp = self.config.affinity_temperature
        exp_affinities = np.exp(affinities / temp)
        normalized = exp_affinities / (exp_affinities.sum() + 1e-8)
        
        return normalized
    
    def get_prototypes(self) -> np.ndarray:
        """
        Get prototype matrix for all liquids.
        
        Returns:
            Matrix of prototypes (num_liquids x embedding_dim)
        """
        prototypes = []
        for liquid_id in sorted(self.liquids.keys()):
            prototypes.append(self.liquids[liquid_id].prototype)
        return np.stack(prototypes)
    
    def record_novelty(self, embedding: np.ndarray, novelty_score: float):
        """
        Record a novelty observation for potential spawning.
        
        Args:
            embedding: Embedding of the novel observation
            novelty_score: How novel this observation is
        """
        self.novelty_buffer.append((embedding.copy(), novelty_score))
        if len(self.novelty_buffer) > self.novelty_buffer_size:
            self.novelty_buffer.pop(0)
    
    def check_spawn(self) -> Optional[Liquid]:
        """
        Check if a new liquid should be spawned.
        
        Returns:
            Newly spawned Liquid or None
        """
        if len(self.liquids) >= self.config.max_liquids:
            return None
        
        if len(self.novelty_buffer) < 10:
            return None
        
        # Find high-novelty embeddings
        high_novelty = [
            (emb, score) for emb, score in self.novelty_buffer
            if score > self.config.spawn_threshold
        ]
        
        if len(high_novelty) < 5:
            return None
        
        # Compute centroid of high-novelty embeddings
        embeddings = np.stack([emb for emb, _ in high_novelty])
        centroid = embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        
        # Check if centroid is far from existing prototypes
        max_similarity = 0.0
        for liquid in self.liquids.values():
            sim = np.dot(centroid, liquid.prototype)
            max_similarity = max(max_similarity, sim)
        
        if max_similarity < self.config.merge_threshold:
            # Spawn new liquid
            new_liquid = self._spawn_liquid(centroid)
            
            # Clear novelty buffer
            self.novelty_buffer = [
                (emb, score) for emb, score in self.novelty_buffer
                if score <= self.config.spawn_threshold
            ]
            
            return new_liquid
        
        return None
    
    def check_merge(self) -> Optional[Tuple[int, int]]:
        """
        Check if any liquids should be merged.
        
        Returns:
            Tuple of (source_id, target_id) to merge, or None
        """
        if len(self.liquids) <= self.config.min_liquids:
            return None
        
        # Find most similar pair
        liquid_ids = sorted(self.liquids.keys())
        best_pair = None
        best_similarity = 0.0
        
        for i, id_i in enumerate(liquid_ids):
            for id_j in liquid_ids[i+1:]:
                liquid_i = self.liquids[id_i]
                liquid_j = self.liquids[id_j]
                
                # Skip if either is nascent or already merging
                if liquid_i.state in [LiquidState.NASCENT, LiquidState.MERGING]:
                    continue
                if liquid_j.state in [LiquidState.NASCENT, LiquidState.MERGING]:
                    continue
                
                similarity = np.dot(liquid_i.prototype, liquid_j.prototype)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (id_i, id_j)
        
        if best_pair and best_similarity > self.config.merge_threshold:
            return best_pair
        
        return None
    
    def merge_liquids(self, source_id: int, target_id: int):
        """
        Merge source liquid into target liquid.
        
        Args:
            source_id: ID of liquid to merge from
            target_id: ID of liquid to merge into
        """
        source = self.liquids[source_id]
        target = self.liquids[target_id]
        
        # Mark source as merging
        source.state = LiquidState.MERGING
        
        # Combine prototypes (weighted by activation count)
        total_activations = source.stats.activation_count + target.stats.activation_count
        if total_activations > 0:
            w_source = source.stats.activation_count / total_activations
            w_target = target.stats.activation_count / total_activations
            target.prototype = w_source * source.prototype + w_target * target.prototype
            target.prototype /= np.linalg.norm(target.prototype) + 1e-8
        
        # Combine statistics
        target.stats.activation_count += source.stats.activation_count
        target.stats.total_reward += source.stats.total_reward
        target.member_columns.update(source.member_columns)
        
        # Combine permissions (union)
        target.permissions.update(source.permissions)
        
        # Remove source
        del self.liquids[source_id]
        
        self.events.append({
            'type': 'merge',
            'source_id': source_id,
            'target_id': target_id,
            'tick': self.current_tick
        })
    
    def check_prune(self) -> List[int]:
        """
        Check which liquids should be pruned.
        
        Returns:
            List of liquid IDs to prune
        """
        if len(self.liquids) <= self.config.min_liquids:
            return []
        
        to_prune = []
        for liquid_id, liquid in self.liquids.items():
            if liquid.should_prune(self.current_tick):
                to_prune.append(liquid_id)
        
        # Don't prune below minimum
        max_to_prune = len(self.liquids) - self.config.min_liquids
        return to_prune[:max_to_prune]
    
    def prune_liquids(self, liquid_ids: List[int]):
        """
        Prune specified liquids.
        
        Args:
            liquid_ids: IDs of liquids to prune
        """
        for liquid_id in liquid_ids:
            if liquid_id in self.liquids:
                self.liquids[liquid_id].state = LiquidState.PRUNING
                del self.liquids[liquid_id]
                
                self.events.append({
                    'type': 'prune',
                    'liquid_id': liquid_id,
                    'tick': self.current_tick
                })
    
    def structure_tick(self):
        """
        Perform periodic structure maintenance.
        
        This is called at regular intervals to:
        - Check for spawning opportunities
        - Check for merging opportunities
        - Prune unused liquids
        - Update liquid states
        """
        self.current_tick += 1
        
        # Check dormancy
        for liquid in self.liquids.values():
            liquid.check_dormancy(self.current_tick)
        
        # Check spawn
        self.check_spawn()
        
        # Check merge
        merge_pair = self.check_merge()
        if merge_pair:
            self.merge_liquids(*merge_pair)
        
        # Check prune
        to_prune = self.check_prune()
        if to_prune:
            self.prune_liquids(to_prune)
    
    def update_member_assignments(self, column_affinities: Dict[int, np.ndarray]):
        """
        Update which columns are members of which liquids.
        
        Args:
            column_affinities: Dict mapping column_id to affinity vector
        """
        # Clear current assignments
        for liquid in self.liquids.values():
            liquid.member_columns.clear()
        
        # Assign columns to their dominant liquid
        liquid_ids = sorted(self.liquids.keys())
        for col_id, affinities in column_affinities.items():
            if len(affinities) > 0:
                dominant_idx = np.argmax(affinities)
                if dominant_idx < len(liquid_ids):
                    dominant_id = liquid_ids[dominant_idx]
                    self.liquids[dominant_id].member_columns.add(col_id)
        
        # Update stats
        for liquid in self.liquids.values():
            liquid.stats.member_count = len(liquid.member_columns)
    
    @property
    def num_liquids(self) -> int:
        """Number of active liquids."""
        return len(self.liquids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the liquid pool."""
        states = defaultdict(int)
        for liquid in self.liquids.values():
            states[liquid.state.value] += 1
        
        return {
            'num_liquids': self.num_liquids,
            'states': dict(states),
            'total_activations': sum(l.stats.activation_count for l in self.liquids.values()),
            'events': len(self.events),
            'current_tick': self.current_tick
        }
    
    def __getitem__(self, liquid_id: int) -> Liquid:
        """Get liquid by ID."""
        return self.liquids[liquid_id]
    
    def __len__(self) -> int:
        """Number of liquids."""
        return len(self.liquids)
    
    def __iter__(self):
        """Iterate over liquids."""
        return iter(self.liquids.values())
    
    def __repr__(self) -> str:
        return f"LiquidPool(num_liquids={self.num_liquids}, tick={self.current_tick})"


if __name__ == "__main__":
    print("=== Liquid System Tests ===\n")
    
    # Create config
    config = LiquidConfig(
        embedding_dim=32,
        initial_num_liquids=4,
        max_liquids=8,
        min_liquids=2
    )
    
    # Create pool
    pool = LiquidPool(config)
    print(f"Created: {pool}")
    print(f"Stats: {pool.get_stats()}\n")
    
    # Test affinity computation
    embedding = np.random.randn(32).astype(np.float32)
    affinities = pool.compute_affinities(embedding)
    print(f"Affinities for random embedding: {affinities}")
    print(f"Sum: {affinities.sum():.4f}\n")
    
    # Simulate some activations
    print("Simulating activations...")
    for i in range(100):
        # Random embedding
        emb = np.random.randn(32).astype(np.float32)
        affinities = pool.compute_affinities(emb)
        
        # Activate dominant liquid
        dominant_id = list(pool.liquids.keys())[np.argmax(affinities)]
        pool.liquids[dominant_id].record_activation(i, reward=np.random.random())
        pool.liquids[dominant_id].update_prototype(emb, weight=affinities.max())
        
        # Record novelty
        novelty = 1.0 - affinities.max()
        pool.record_novelty(emb, novelty)
        
        # Periodic structure tick
        if i % 10 == 0:
            pool.structure_tick()
    
    print(f"\nAfter 100 iterations:")
    print(f"Stats: {pool.get_stats()}")
    print(f"\nLiquids:")
    for liquid in pool:
        print(f"  {liquid}")
        print(f"    Info: {liquid.get_info()}")
