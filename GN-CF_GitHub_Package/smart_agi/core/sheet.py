"""
Sheet (GyrusNet) Module

The Sheet is a 2D grid of microcolumns with Local Excitation / Lateral Inhibition
(LE/LI) connectivity. This creates a self-organizing structure where:
- Nearby columns tend to activate together (local excitation)
- Distant columns inhibit each other (lateral inhibition)
- Modules naturally emerge through Hebbian learning

Inspired by cortical organization and Turing pattern formation.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict

from .microcolumn import Microcolumn, MicrocolumnConfig, MicrocolumnFactory, ColumnMode
from .ternary import quantize_to_ternary


@dataclass
class SheetConfig:
    """Configuration for the GyrusNet sheet."""
    size: int = 8                    # Grid size (size x size)
    input_dim: int = 64              # Input dimension from workspace
    hidden_dim: int = 32             # Hidden dimension per column
    output_dim: int = 16             # Output dimension per column
    state_dim: int = 16              # State dimension per column
    num_liquids: int = 8             # Number of liquids
    initial_sparsity: float = 0.5    # Initial weight sparsity
    
    # LE/LI connectivity parameters (Mexican hat profile)
    sigma_excitation: float = 1.2    # Excitation radius (narrow)
    sigma_inhibition: float = 3.0    # Inhibition radius (wider)
    excitation_strength: float = 1.0  # Strength of excitation Gaussian
    inhibition_strength: float = 0.5  # Strength of inhibition Gaussian
    excitation_threshold: float = 0.15  # Threshold for +1 connection
    inhibition_threshold: float = 0.05  # Threshold for -1 connection
    
    # Hebbian parameters
    hebbian_learning_rate: float = 0.05
    lateral_learning_rate: float = 0.02


class Sheet:
    """
    A 2D sheet of microcolumns with LE/LI connectivity.
    
    The sheet implements:
    - Local Excitation: Nearby columns strengthen each other
    - Lateral Inhibition: Distant columns suppress each other
    - Hebbian Learning: Connections strengthen based on co-activation
    - Module Formation: Clusters of columns specialize together
    
    Connectivity follows a Mexican hat profile:
        w(d) = exp(-d²/2σ_e²) - exp(-d²/2σ_i²)
    
    where d is distance, σ_e is excitation radius, σ_i is inhibition radius.
    """
    
    def __init__(self, config: SheetConfig):
        """
        Initialize the sheet.
        
        Args:
            config: Sheet configuration
        """
        self.config = config
        self.size = config.size
        self.num_columns = config.size ** 2
        
        # Create microcolumns
        col_config = MicrocolumnConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            state_dim=config.state_dim,
            num_liquids=config.num_liquids,
            initial_sparsity=config.initial_sparsity
        )
        
        factory = MicrocolumnFactory(col_config)
        self.columns = factory.create_grid(config.size)
        
        # Create position lookup
        self.pos_to_idx = {}
        for idx, col in enumerate(self.columns):
            self.pos_to_idx[col.position] = idx
        
        # Initialize LE/LI connectivity matrix
        self.lateral_weights = self._init_lateral_connectivity()
        
        # Track activations for Hebbian learning
        self.last_activations = np.zeros(self.num_columns, dtype=np.float32)
        self.activation_history = []
        self.history_size = 20
        
        # Module tracking
        self.module_assignments = np.zeros(self.num_columns, dtype=np.int32)
        
    def _init_lateral_connectivity(self) -> np.ndarray:
        """
        Initialize lateral connectivity matrix using Mexican hat profile.
        
        Returns:
            Ternary connectivity matrix (num_columns x num_columns)
        """
        n = self.num_columns
        weights = np.zeros((n, n), dtype=np.int8)
        
        sigma_e = self.config.sigma_excitation
        sigma_i = self.config.sigma_inhibition
        exc_strength = self.config.excitation_strength
        inh_strength = self.config.inhibition_strength
        exc_thresh = self.config.excitation_threshold
        inh_thresh = self.config.inhibition_threshold
        
        for i in range(n):
            pos_i = self.columns[i].position
            for j in range(n):
                if i == j:
                    continue  # No self-connection
                    
                pos_j = self.columns[j].position
                
                # Compute Euclidean distance
                d = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                # Mexican hat: local excitation, lateral inhibition
                # Use different strengths for excitation and inhibition
                excitation = exc_strength * np.exp(-d**2 / (2 * sigma_e**2))
                inhibition = inh_strength * np.exp(-d**2 / (2 * sigma_i**2))
                raw_weight = excitation - inhibition
                
                # Quantize to ternary
                if raw_weight > exc_thresh:
                    weights[i, j] = 1   # Excitation
                elif raw_weight < -inh_thresh:
                    weights[i, j] = -1  # Inhibition
                # else: 0 (no connection)
        
        return weights
    
    def get_neighbors(self, idx: int, connection_type: str = 'all') -> List[int]:
        """
        Get indices of connected neighbors.
        
        Args:
            idx: Column index
            connection_type: 'excitatory', 'inhibitory', or 'all'
        
        Returns:
            List of neighbor indices
        """
        if connection_type == 'excitatory':
            return list(np.where(self.lateral_weights[idx] == 1)[0])
        elif connection_type == 'inhibitory':
            return list(np.where(self.lateral_weights[idx] == -1)[0])
        else:
            return list(np.where(self.lateral_weights[idx] != 0)[0])
    
    def forward(self, workspace_input: np.ndarray, 
                selected_indices: Optional[List[int]] = None,
                mode: int = ColumnMode.EXPERIENCER) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input through selected columns.
        
        Args:
            workspace_input: Input from workspace (input_dim,)
            selected_indices: Indices of columns to activate (None = all)
            mode: Operating mode for all columns
        
        Returns:
            Tuple of (outputs, activations)
            - outputs: (num_selected, output_dim) array of column outputs
            - activations: (num_columns,) array of activation levels
        """
        if selected_indices is None:
            selected_indices = list(range(self.num_columns))
        
        # Compute raw outputs from selected columns
        outputs = []
        raw_activations = np.zeros(self.num_columns, dtype=np.float32)
        
        for idx in selected_indices:
            col = self.columns[idx]
            output = col.forward(workspace_input, mode=mode)
            outputs.append(output)
            raw_activations[idx] = np.mean(np.abs(output))
        
        # Apply lateral interactions
        lateral_input = self.lateral_weights @ raw_activations
        
        # Modulate activations
        modulated_activations = raw_activations + 0.1 * lateral_input
        modulated_activations = np.clip(modulated_activations, -2, 2)
        
        # Store for Hebbian learning
        self.last_activations = modulated_activations
        self._update_activation_history(modulated_activations)
        
        return np.array(outputs), modulated_activations
    
    def _update_activation_history(self, activations: np.ndarray):
        """Update activation history for Hebbian learning."""
        self.activation_history.append(activations.copy())
        if len(self.activation_history) > self.history_size:
            self.activation_history.pop(0)
    
    def hebbian_lateral_update(self) -> np.ndarray:
        """
        Apply Hebbian learning to lateral connections.
        
        Strengthens connections between columns that co-activate,
        weakens connections between columns that anti-correlate.
        
        Returns:
            Matrix of weight updates applied
        """
        if len(self.activation_history) < 5:
            return np.zeros_like(self.lateral_weights)
        
        # Stack activation history
        history = np.stack(self.activation_history)  # (T, num_columns)
        
        # Compute correlation matrix
        # Normalize activations
        centered = history - history.mean(axis=0, keepdims=True)
        std = centered.std(axis=0, keepdims=True) + 1e-8
        normalized = centered / std
        
        # Correlation
        correlation = (normalized.T @ normalized) / len(history)
        
        # Convert to ternary updates
        lr = self.config.lateral_learning_rate
        threshold = 0.3
        
        updates = np.where(correlation > threshold, 1,
                          np.where(correlation < -threshold, -1, 0))
        
        # Stochastic application
        mask = np.random.random(self.lateral_weights.shape) < lr
        updates = updates * mask
        
        # Don't update self-connections
        np.fill_diagonal(updates, 0)
        
        # Apply updates
        self.lateral_weights = np.clip(
            self.lateral_weights + updates, -1, 1
        ).astype(np.int8)
        
        return updates.astype(np.int8)
    
    def hebbian_column_update(self, workspace_input: np.ndarray):
        """
        Apply Hebbian learning to individual columns.
        
        Args:
            workspace_input: Input that produced current activations
        """
        lr = self.config.hebbian_learning_rate
        
        for idx, col in enumerate(self.columns):
            if self.last_activations[idx] != 0:
                col.hebbian_update(workspace_input, learning_rate=lr)
    
    def compute_modularity(self) -> float:
        """
        Compute modularity score of the current lateral connectivity.
        
        Higher modularity means clearer module boundaries.
        
        Returns:
            Modularity score (0 to 1)
        """
        # Assign modules based on dominant liquid
        for idx, col in enumerate(self.columns):
            self.module_assignments[idx] = col.dominant_liquid
        
        # Compute modularity (fraction of within-module excitatory connections)
        total_excitatory = 0
        within_module_excitatory = 0
        
        for i in range(self.num_columns):
            for j in range(self.num_columns):
                if self.lateral_weights[i, j] == 1:
                    total_excitatory += 1
                    if self.module_assignments[i] == self.module_assignments[j]:
                        within_module_excitatory += 1
        
        if total_excitatory == 0:
            return 0.0
        
        return within_module_excitatory / total_excitatory
    
    def compute_intra_module_strength(self) -> float:
        """
        Compute average connection strength within modules.
        
        Returns:
            Average intra-module connection strength
        """
        strengths = []
        
        for i in range(self.num_columns):
            module_i = self.module_assignments[i]
            for j in range(self.num_columns):
                if i != j and self.module_assignments[j] == module_i:
                    strengths.append(self.lateral_weights[i, j])
        
        if len(strengths) == 0:
            return 0.0
        
        return np.mean(strengths)
    
    def compute_inter_module_inhibition(self) -> float:
        """
        Compute average connection strength between modules.
        
        Returns:
            Average inter-module connection strength (should be negative)
        """
        strengths = []
        
        for i in range(self.num_columns):
            module_i = self.module_assignments[i]
            for j in range(self.num_columns):
                if i != j and self.module_assignments[j] != module_i:
                    strengths.append(self.lateral_weights[i, j])
        
        if len(strengths) == 0:
            return 0.0
        
        return np.mean(strengths)
    
    def get_module_map(self) -> np.ndarray:
        """
        Get 2D map of module assignments.
        
        Returns:
            2D array (size x size) of module indices
        """
        module_map = np.zeros((self.size, self.size), dtype=np.int32)
        
        for idx, col in enumerate(self.columns):
            x, y = col.position
            module_map[y, x] = col.dominant_liquid
        
        return module_map
    
    def get_activation_map(self) -> np.ndarray:
        """
        Get 2D map of current activations.
        
        Returns:
            2D array (size x size) of activation levels
        """
        activation_map = np.zeros((self.size, self.size), dtype=np.float32)
        
        for idx, col in enumerate(self.columns):
            x, y = col.position
            activation_map[y, x] = self.last_activations[idx]
        
        return activation_map
    
    def get_connectivity_stats(self) -> Dict[str, Any]:
        """Get statistics about lateral connectivity."""
        weights = self.lateral_weights
        
        return {
            'excitatory_fraction': np.mean(weights == 1),
            'inhibitory_fraction': np.mean(weights == -1),
            'zero_fraction': np.mean(weights == 0),
            'modularity': self.compute_modularity(),
            'intra_module_strength': self.compute_intra_module_strength(),
            'inter_module_inhibition': self.compute_inter_module_inhibition()
        }
    
    def reset_states(self):
        """Reset all column states."""
        for col in self.columns:
            col.reset_state()
        self.last_activations = np.zeros(self.num_columns, dtype=np.float32)
        self.activation_history = []
    
    def __getitem__(self, idx: int) -> Microcolumn:
        """Get column by index."""
        return self.columns[idx]
    
    def __len__(self) -> int:
        """Number of columns."""
        return self.num_columns
    
    def __repr__(self) -> str:
        stats = self.get_connectivity_stats()
        return (f"Sheet(size={self.size}x{self.size}, "
                f"columns={self.num_columns}, "
                f"modularity={stats['modularity']:.3f})")


class SheetVisualizer:
    """Utility class for visualizing sheet state."""
    
    @staticmethod
    def print_module_map(sheet: Sheet):
        """Print ASCII visualization of module assignments."""
        module_map = sheet.get_module_map()
        symbols = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        print("Module Map:")
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
        for y in range(sheet.size):
            row = "| "
            for x in range(sheet.size):
                module = module_map[y, x]
                row += symbols[module % len(symbols)] + " "
            row += "|"
            print(row)
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
    
    @staticmethod
    def print_activation_map(sheet: Sheet):
        """Print ASCII visualization of activations."""
        activation_map = sheet.get_activation_map()
        
        print("Activation Map:")
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
        for y in range(sheet.size):
            row = "| "
            for x in range(sheet.size):
                act = activation_map[y, x]
                if act > 1:
                    row += "█ "
                elif act > 0.5:
                    row += "▓ "
                elif act > 0:
                    row += "░ "
                elif act < -0.5:
                    row += "▒ "
                else:
                    row += "· "
            row += "|"
            print(row)
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
    
    @staticmethod
    def print_connectivity_pattern(sheet: Sheet, idx: int):
        """Print connectivity pattern for a single column."""
        pos = sheet.columns[idx].position
        
        print(f"Connectivity for column {idx} at {pos}:")
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
        
        for y in range(sheet.size):
            row = "| "
            for x in range(sheet.size):
                other_idx = sheet.pos_to_idx.get((x, y))
                if other_idx is None:
                    row += "? "
                elif other_idx == idx:
                    row += "● "
                else:
                    w = sheet.lateral_weights[idx, other_idx]
                    if w == 1:
                        row += "+ "
                    elif w == -1:
                        row += "- "
                    else:
                        row += "· "
            row += "|"
            print(row)
        print("+" + "-" * (sheet.size * 2 + 1) + "+")
        print("Legend: ● = self, + = excitatory, - = inhibitory, · = none")


if __name__ == "__main__":
    print("=== Sheet (GyrusNet) Tests ===\n")
    
    # Create sheet
    config = SheetConfig(
        size=6,
        input_dim=32,
        hidden_dim=16,
        output_dim=8,
        state_dim=8,
        num_liquids=4,
        sigma_excitation=1.5,
        sigma_inhibition=4.0
    )
    
    sheet = Sheet(config)
    print(f"Created: {sheet}\n")
    
    # Show initial connectivity
    print("Connectivity stats:", sheet.get_connectivity_stats())
    print()
    
    # Visualize connectivity for center column
    center_idx = sheet.size * (sheet.size // 2) + (sheet.size // 2)
    SheetVisualizer.print_connectivity_pattern(sheet, center_idx)
    print()
    
    # Test forward pass
    x = np.random.randn(32).astype(np.float32)
    outputs, activations = sheet.forward(x, selected_indices=[0, 1, 2, 10, 20])
    print(f"Forward pass: outputs shape={outputs.shape}")
    print(f"Activations: min={activations.min():.3f}, max={activations.max():.3f}")
    print()
    
    # Show activation map
    SheetVisualizer.print_activation_map(sheet)
    print()
    
    # Run several iterations to build up activation history
    print("Running 20 iterations...")
    for i in range(20):
        x = np.random.randn(32).astype(np.float32)
        # Activate different subsets
        selected = np.random.choice(sheet.num_columns, size=10, replace=False).tolist()
        outputs, activations = sheet.forward(x, selected_indices=selected)
    
    # Apply Hebbian update
    updates = sheet.hebbian_lateral_update()
    print(f"Hebbian updates: {np.sum(updates != 0)} connections modified")
    print()
    
    # Show module map
    SheetVisualizer.print_module_map(sheet)
    print()
    
    # Final stats
    print("Final connectivity stats:", sheet.get_connectivity_stats())
