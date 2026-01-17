"""
Router Module (Thalamic Router)

The Router selects which microcolumns to activate for each input.
It implements:
- Top-K selection with load balancing
- Cost-aware routing (prefer cheaper columns)
- Hebbian coupling (columns that work together get routed together)
- Modular team formation (groups of columns that specialize together)

Inspired by the thalamus, which routes information to appropriate cortical regions.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .ternary import quantize_to_ternary, TernaryArray


@dataclass
class RouterConfig:
    """Configuration for the router."""
    num_columns: int = 64           # Number of columns to route to
    embedding_dim: int = 64         # Dimension of query/key embeddings
    top_k: int = 8                  # Number of columns to select
    
    # Cost parameters
    wire_cost_weight: float = 0.1   # Weight for wiring cost
    compute_cost_weight: float = 0.05  # Weight for compute cost
    
    # Load balancing
    balance_weight: float = 0.1     # Weight for load balancing term
    
    # Hebbian coupling
    coupling_learning_rate: float = 0.01
    coupling_decay: float = 0.99    # Decay for coupling weights
    coupling_threshold: float = 0.5  # Threshold for strong coupling
    
    # Team formation
    team_size: int = 4              # Target team size
    team_cohesion_bonus: float = 0.1  # Bonus for selecting team members


class HebbianCouplingMatrix:
    """
    Tracks co-activation patterns between columns using Hebbian learning.
    
    Columns that are frequently activated together develop strong coupling,
    which encourages them to be selected together in the future.
    """
    
    def __init__(self, num_columns: int, learning_rate: float = 0.01,
                 decay: float = 0.99):
        """
        Initialize coupling matrix.
        
        Args:
            num_columns: Number of columns
            learning_rate: Learning rate for updates
            decay: Decay factor applied each step
        """
        self.num_columns = num_columns
        self.learning_rate = learning_rate
        self.decay = decay
        
        # Coupling matrix (symmetric, diagonal is 0)
        self.coupling = np.zeros((num_columns, num_columns), dtype=np.float32)
        
        # Activation counts for normalization
        self.activation_counts = np.zeros(num_columns, dtype=np.int64)
        self.co_activation_counts = np.zeros((num_columns, num_columns), dtype=np.int64)
        
    def update(self, activated_indices: List[int]):
        """
        Update coupling based on co-activation.
        
        Args:
            activated_indices: Indices of columns that were activated together
        """
        # Apply decay
        self.coupling *= self.decay
        
        # Update activation counts
        for idx in activated_indices:
            self.activation_counts[idx] += 1
        
        # Update co-activation counts and coupling
        for i, idx_i in enumerate(activated_indices):
            for idx_j in activated_indices[i+1:]:
                self.co_activation_counts[idx_i, idx_j] += 1
                self.co_activation_counts[idx_j, idx_i] += 1
                
                # Hebbian update: strengthen coupling
                self.coupling[idx_i, idx_j] += self.learning_rate
                self.coupling[idx_j, idx_i] += self.learning_rate
        
        # Clamp to [0, 1]
        self.coupling = np.clip(self.coupling, 0, 1)
    
    def get_coupled_columns(self, idx: int, threshold: float = 0.5) -> List[int]:
        """
        Get columns strongly coupled to a given column.
        
        Args:
            idx: Column index
            threshold: Coupling threshold
        
        Returns:
            List of coupled column indices
        """
        coupled = np.where(self.coupling[idx] > threshold)[0]
        return [int(c) for c in coupled if c != idx]
    
    def get_coupling_strength(self, idx_i: int, idx_j: int) -> float:
        """Get coupling strength between two columns."""
        return float(self.coupling[idx_i, idx_j])
    
    def get_team_bonus(self, selected_indices: List[int]) -> float:
        """
        Compute bonus for selecting a cohesive team.
        
        Args:
            selected_indices: Currently selected indices
        
        Returns:
            Average coupling among selected columns
        """
        if len(selected_indices) < 2:
            return 0.0
        
        total_coupling = 0.0
        count = 0
        
        for i, idx_i in enumerate(selected_indices):
            for idx_j in selected_indices[i+1:]:
                total_coupling += self.coupling[idx_i, idx_j]
                count += 1
        
        return total_coupling / count if count > 0 else 0.0
    
    def identify_teams(self, threshold: float = 0.5) -> List[Set[int]]:
        """
        Identify teams of strongly coupled columns.
        
        Args:
            threshold: Coupling threshold for team membership
        
        Returns:
            List of teams (sets of column indices)
        """
        visited = set()
        teams = []
        
        for idx in range(self.num_columns):
            if idx in visited:
                continue
            
            # BFS to find connected component
            team = {idx}
            queue = [idx]
            
            while queue:
                current = queue.pop(0)
                coupled = self.get_coupled_columns(current, threshold)
                
                for c in coupled:
                    if c not in visited:
                        visited.add(c)
                        team.add(c)
                        queue.append(c)
            
            visited.add(idx)
            if len(team) > 1:
                teams.append(team)
        
        return teams
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about coupling."""
        non_zero = self.coupling > 0.01
        
        return {
            'mean_coupling': float(self.coupling[non_zero].mean()) if non_zero.any() else 0.0,
            'max_coupling': float(self.coupling.max()),
            'num_strong_pairs': int((self.coupling > 0.5).sum() // 2),
            'num_teams': len(self.identify_teams())
        }


class Router:
    """
    Thalamic Router for selecting microcolumns.
    
    The router:
    1. Computes relevance scores for each column
    2. Applies cost penalties (wiring, compute)
    3. Adds Hebbian coupling bonuses
    4. Selects top-K columns
    5. Returns selection weights
    """
    
    def __init__(self, config: RouterConfig):
        """
        Initialize the router.
        
        Args:
            config: Router configuration
        """
        self.config = config
        
        # Key projection for each column (ternary weights)
        self.keys = TernaryArray.random(
            (config.num_columns, config.embedding_dim),
            sparsity=0.3
        )
        
        # Column costs (compute cost based on column complexity)
        self.compute_costs = np.random.uniform(0.5, 1.5, config.num_columns).astype(np.float32)
        
        # Wire costs (based on position - computed externally)
        self.wire_costs = np.zeros(config.num_columns, dtype=np.float32)
        
        # Load balancing: track usage
        self.usage_counts = np.zeros(config.num_columns, dtype=np.int64)
        self.total_selections = 0
        
        # Hebbian coupling
        self.coupling = HebbianCouplingMatrix(
            config.num_columns,
            learning_rate=config.coupling_learning_rate,
            decay=config.coupling_decay
        )
        
        # Selection history
        self.selection_history: List[List[int]] = []
        self.history_size = 100
        
    def set_wire_costs(self, positions: List[Tuple[int, int]], 
                       reference_pos: Tuple[int, int] = (0, 0)):
        """
        Set wire costs based on column positions.
        
        Args:
            positions: List of (x, y) positions for each column
            reference_pos: Reference position (e.g., workspace center)
        """
        for idx, pos in enumerate(positions):
            distance = np.sqrt(
                (pos[0] - reference_pos[0])**2 + 
                (pos[1] - reference_pos[1])**2
            )
            self.wire_costs[idx] = distance
        
        # Normalize to [0, 1]
        if self.wire_costs.max() > 0:
            self.wire_costs /= self.wire_costs.max()
    
    def compute_scores(self, query: np.ndarray, 
                       current_selection: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute selection scores for all columns.
        
        Args:
            query: Query vector from workspace (embedding_dim,)
            current_selection: Already selected columns (for team bonus)
        
        Returns:
            Score for each column
        """
        # Base relevance: dot product with keys
        relevance = self.keys.dot(query)
        
        # Cost penalties
        wire_penalty = self.config.wire_cost_weight * self.wire_costs
        compute_penalty = self.config.compute_cost_weight * self.compute_costs
        
        # Load balancing penalty
        if self.total_selections > 0:
            expected_usage = self.total_selections / self.config.num_columns
            usage_ratio = self.usage_counts / (expected_usage + 1e-8)
            balance_penalty = self.config.balance_weight * np.log1p(usage_ratio)
        else:
            balance_penalty = np.zeros(self.config.num_columns)
        
        # Hebbian coupling bonus
        coupling_bonus = np.zeros(self.config.num_columns, dtype=np.float32)
        if current_selection:
            for idx in current_selection:
                coupling_bonus += self.coupling.coupling[idx]
            coupling_bonus /= len(current_selection)
            coupling_bonus *= self.config.team_cohesion_bonus
        
        # Final score
        scores = relevance - wire_penalty - compute_penalty - balance_penalty + coupling_bonus
        
        return scores
    
    def select(self, query: np.ndarray, 
               k: Optional[int] = None) -> Tuple[List[int], np.ndarray]:
        """
        Select top-K columns for the given query.
        
        Args:
            query: Query vector from workspace
            k: Number of columns to select (default: config.top_k)
        
        Returns:
            Tuple of (selected indices, selection weights)
        """
        if k is None:
            k = self.config.top_k
        
        # Iterative selection with coupling bonus
        selected = []
        remaining_k = k
        
        while remaining_k > 0:
            # Compute scores with current selection
            scores = self.compute_scores(query, selected)
            
            # Mask already selected
            for idx in selected:
                scores[idx] = float('-inf')
            
            # Select best
            if remaining_k == 1:
                best_idx = int(np.argmax(scores))
                selected.append(best_idx)
                remaining_k -= 1
            else:
                # Select top remaining
                top_indices = np.argsort(scores)[-remaining_k:][::-1]
                selected.extend([int(i) for i in top_indices])
                remaining_k = 0
        
        # Compute final weights (softmax of scores)
        final_scores = self.compute_scores(query, selected)
        selected_scores = final_scores[selected]
        weights = np.exp(selected_scores - selected_scores.max())
        weights /= weights.sum() + 1e-8
        
        # Update usage counts
        for idx in selected:
            self.usage_counts[idx] += 1
        self.total_selections += 1
        
        # Update Hebbian coupling
        self.coupling.update(selected)
        
        # Store in history
        self.selection_history.append(selected)
        if len(self.selection_history) > self.history_size:
            self.selection_history.pop(0)
        
        return selected, weights.astype(np.float32)
    
    def update_keys(self, column_embeddings: np.ndarray, learning_rate: float = 0.01):
        """
        Update key projections based on column embeddings.
        
        Args:
            column_embeddings: Current embeddings for all columns
            learning_rate: Learning rate for update
        """
        # Quantize embeddings to ternary
        quantized = np.where(column_embeddings > 0.33, 1,
                            np.where(column_embeddings < -0.33, -1, 0))
        
        # Blend with current keys
        blended = (1 - learning_rate) * self.keys.data + learning_rate * quantized
        self.keys = TernaryArray.from_continuous(blended, threshold=0.33)
    
    def get_teams(self) -> List[Set[int]]:
        """Get identified teams of coupled columns."""
        return self.coupling.identify_teams(self.config.coupling_threshold)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        # Selection entropy
        if self.total_selections > 0:
            probs = self.usage_counts / self.total_selections
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0.0
        
        # Dead columns (never selected)
        dead_fraction = np.mean(self.usage_counts == 0)
        
        # Top-1 share (fraction of selections going to most used column)
        if self.total_selections > 0:
            top1_share = self.usage_counts.max() / self.total_selections
        else:
            top1_share = 0.0
        
        return {
            'total_selections': self.total_selections,
            'selection_entropy': float(entropy),
            'dead_column_fraction': float(dead_fraction),
            'top1_share': float(top1_share),
            'coupling_stats': self.coupling.get_stats(),
            'num_teams': len(self.get_teams())
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"Router(columns={self.config.num_columns}, "
                f"top_k={self.config.top_k}, "
                f"teams={stats['num_teams']}, "
                f"entropy={stats['selection_entropy']:.2f})")


if __name__ == "__main__":
    print("=== Router Tests ===\n")
    
    # Create router
    config = RouterConfig(
        num_columns=64,
        embedding_dim=32,
        top_k=8
    )
    
    router = Router(config)
    print(f"Created: {router}\n")
    
    # Set wire costs based on grid positions
    positions = [(i % 8, i // 8) for i in range(64)]
    router.set_wire_costs(positions, reference_pos=(4, 4))
    
    # Test selection
    query = np.random.randn(32).astype(np.float32)
    selected, weights = router.select(query)
    print(f"Selected indices: {selected}")
    print(f"Weights: {weights}\n")
    
    # Run many selections to build up coupling
    print("Running 500 selections...")
    for i in range(500):
        query = np.random.randn(32).astype(np.float32)
        selected, weights = router.select(query)
    
    print(f"\nAfter 500 selections:")
    print(f"Stats: {router.get_stats()}")
    
    # Show teams
    teams = router.get_teams()
    print(f"\nIdentified {len(teams)} teams:")
    for i, team in enumerate(teams[:5]):
        print(f"  Team {i}: {sorted(team)}")
