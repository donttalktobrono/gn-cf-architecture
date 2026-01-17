"""
Dynamic Growth and Pruning Module

This module implements mechanisms for the network to grow and shrink based on
task demands. Key features:
- Growth triggers (high utilization, novelty, task failure)
- Pruning triggers (low activation, redundancy, resource pressure)
- Resource cost model (growth has a cost)
- Neurogenesis (adding new neurons/columns)
- Synaptic pruning (removing weak connections)

This creates a system where the network's structure adapts to task complexity.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class GrowthEvent(Enum):
    """Types of growth events."""
    ADD_NEURON = "add_neuron"
    ADD_COLUMN = "add_column"
    ADD_CONNECTION = "add_connection"
    PRUNE_NEURON = "prune_neuron"
    PRUNE_COLUMN = "prune_column"
    PRUNE_CONNECTION = "prune_connection"


@dataclass
class ResourceCosts:
    """Cost model for network resources."""
    neuron_cost: float = 1.0           # Cost per neuron
    connection_cost: float = 0.1       # Cost per connection
    column_cost: float = 10.0          # Cost per microcolumn
    liquid_cost: float = 20.0          # Cost per liquid
    memory_cost: float = 0.01          # Cost per memory slot
    compute_cost: float = 0.001        # Cost per FLOP equivalent


@dataclass
class GrowthConfig:
    """Configuration for growth/pruning system."""
    # Resource budget
    initial_budget: float = 1000.0
    max_budget: float = 5000.0
    budget_regeneration_rate: float = 1.0  # Budget gained per tick
    
    # Growth triggers
    utilization_growth_threshold: float = 0.9  # Grow if utilization > this
    novelty_growth_threshold: float = 0.8      # Grow if novelty > this
    failure_growth_threshold: float = 0.3      # Grow if failure rate > this
    
    # Pruning triggers
    activation_prune_threshold: int = 10       # Prune if activations < this
    weight_prune_threshold: float = 0.1        # Prune connections weaker than this
    redundancy_prune_threshold: float = 0.95   # Prune if similarity > this
    
    # Growth limits
    max_columns: int = 256
    max_neurons_per_column: int = 128
    min_columns: int = 4
    min_neurons_per_column: int = 8
    
    # Growth rates
    growth_batch_size: int = 4        # Neurons to add at once
    prune_batch_size: int = 4         # Neurons to prune at once
    
    # Costs
    costs: ResourceCosts = field(default_factory=ResourceCosts)


class GrowthMetrics:
    """Tracks metrics relevant to growth decisions."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        
        # Utilization metrics
        self.utilization_history: List[float] = []
        
        # Novelty metrics
        self.novelty_history: List[float] = []
        
        # Task performance metrics
        self.success_history: List[bool] = []
        self.reward_history: List[float] = []
        
        # Resource metrics
        self.compute_history: List[float] = []
        self.memory_history: List[float] = []
        
    def record_utilization(self, utilization: float):
        """Record utilization metric."""
        self.utilization_history.append(utilization)
        if len(self.utilization_history) > self.window_size:
            self.utilization_history.pop(0)
    
    def record_novelty(self, novelty: float):
        """Record novelty metric."""
        self.novelty_history.append(novelty)
        if len(self.novelty_history) > self.window_size:
            self.novelty_history.pop(0)
    
    def record_task_result(self, success: bool, reward: float):
        """Record task result."""
        self.success_history.append(success)
        self.reward_history.append(reward)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
            self.reward_history.pop(0)
    
    def record_resource_usage(self, compute: float, memory: float):
        """Record resource usage."""
        self.compute_history.append(compute)
        self.memory_history.append(memory)
        if len(self.compute_history) > self.window_size:
            self.compute_history.pop(0)
            self.memory_history.pop(0)
    
    @property
    def avg_utilization(self) -> float:
        """Average utilization over window."""
        if not self.utilization_history:
            return 0.0
        return np.mean(self.utilization_history)
    
    @property
    def avg_novelty(self) -> float:
        """Average novelty over window."""
        if not self.novelty_history:
            return 0.0
        return np.mean(self.novelty_history)
    
    @property
    def failure_rate(self) -> float:
        """Failure rate over window."""
        if not self.success_history:
            return 0.0
        return 1.0 - np.mean(self.success_history)
    
    @property
    def avg_reward(self) -> float:
        """Average reward over window."""
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            'avg_utilization': self.avg_utilization,
            'avg_novelty': self.avg_novelty,
            'failure_rate': self.failure_rate,
            'avg_reward': self.avg_reward
        }


class ResourceBudget:
    """Manages resource budget for growth."""
    
    def __init__(self, config: GrowthConfig):
        """
        Initialize resource budget.
        
        Args:
            config: Growth configuration
        """
        self.config = config
        self.current_budget = config.initial_budget
        self.total_spent = 0.0
        self.spending_history: List[Tuple[str, float]] = []
        
    def can_afford(self, cost: float) -> bool:
        """Check if we can afford a cost."""
        return self.current_budget >= cost
    
    def spend(self, cost: float, description: str) -> bool:
        """
        Spend from budget.
        
        Args:
            cost: Amount to spend
            description: What the spending is for
        
        Returns:
            True if successful, False if insufficient budget
        """
        if not self.can_afford(cost):
            return False
        
        self.current_budget -= cost
        self.total_spent += cost
        self.spending_history.append((description, cost))
        
        return True
    
    def regenerate(self, ticks: int = 1):
        """
        Regenerate budget over time.
        
        Args:
            ticks: Number of ticks to regenerate for
        """
        regen = self.config.budget_regeneration_rate * ticks
        self.current_budget = min(
            self.current_budget + regen,
            self.config.max_budget
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        return {
            'current_budget': self.current_budget,
            'total_spent': self.total_spent,
            'budget_utilization': self.total_spent / (self.total_spent + self.current_budget + 1e-8)
        }


class GrowthController:
    """
    Controls network growth and pruning.
    
    Makes decisions about when to grow/prune based on metrics and budget.
    """
    
    def __init__(self, config: GrowthConfig):
        """
        Initialize growth controller.
        
        Args:
            config: Growth configuration
        """
        self.config = config
        self.metrics = GrowthMetrics()
        self.budget = ResourceBudget(config)
        
        # Track current network size
        self.num_columns = 0
        self.neurons_per_column: Dict[int, int] = {}
        self.total_connections = 0
        
        # Event log
        self.events: List[Dict[str, Any]] = []
        self.current_tick = 0
        
    def set_network_size(self, num_columns: int, 
                         neurons_per_column: Dict[int, int],
                         total_connections: int):
        """
        Update tracked network size.
        
        Args:
            num_columns: Number of columns
            neurons_per_column: Dict mapping column_id to neuron count
            total_connections: Total number of connections
        """
        self.num_columns = num_columns
        self.neurons_per_column = neurons_per_column.copy()
        self.total_connections = total_connections
    
    def compute_current_cost(self) -> float:
        """Compute current total resource cost of network."""
        costs = self.config.costs
        
        total = 0.0
        total += costs.column_cost * self.num_columns
        total += costs.neuron_cost * sum(self.neurons_per_column.values())
        total += costs.connection_cost * self.total_connections
        
        return total
    
    def should_grow(self) -> Tuple[bool, str]:
        """
        Determine if network should grow.
        
        Returns:
            Tuple of (should_grow, reason)
        """
        # Check budget
        min_growth_cost = self.config.costs.neuron_cost * self.config.growth_batch_size
        if not self.budget.can_afford(min_growth_cost):
            return False, "insufficient_budget"
        
        # Check limits
        if self.num_columns >= self.config.max_columns:
            return False, "max_columns_reached"
        
        # Check triggers
        summary = self.metrics.get_summary()
        
        if summary['avg_utilization'] > self.config.utilization_growth_threshold:
            return True, "high_utilization"
        
        if summary['avg_novelty'] > self.config.novelty_growth_threshold:
            return True, "high_novelty"
        
        if summary['failure_rate'] > self.config.failure_growth_threshold:
            return True, "high_failure_rate"
        
        return False, "no_trigger"
    
    def should_prune(self) -> Tuple[bool, str]:
        """
        Determine if network should prune.
        
        Returns:
            Tuple of (should_prune, reason)
        """
        # Check limits
        if self.num_columns <= self.config.min_columns:
            return False, "min_columns_reached"
        
        # Check if we're under resource pressure
        current_cost = self.compute_current_cost()
        if current_cost > self.budget.current_budget * 2:
            return True, "resource_pressure"
        
        # Check metrics - prune if doing well with excess capacity
        summary = self.metrics.get_summary()
        
        if summary['avg_utilization'] < 0.3 and summary['failure_rate'] < 0.1:
            return True, "excess_capacity"
        
        return False, "no_trigger"
    
    def plan_growth(self) -> List[Dict[str, Any]]:
        """
        Plan growth operations.
        
        Returns:
            List of growth operations to perform
        """
        should_grow, reason = self.should_grow()
        if not should_grow:
            return []
        
        operations = []
        costs = self.config.costs
        
        # Decide what to grow
        if reason == "high_utilization":
            # Add neurons to existing columns
            for col_id, neuron_count in self.neurons_per_column.items():
                if neuron_count < self.config.max_neurons_per_column:
                    cost = costs.neuron_cost * self.config.growth_batch_size
                    if self.budget.can_afford(cost):
                        operations.append({
                            'type': GrowthEvent.ADD_NEURON,
                            'column_id': col_id,
                            'count': self.config.growth_batch_size,
                            'cost': cost
                        })
                        break
        
        elif reason in ["high_novelty", "high_failure_rate"]:
            # Add new column
            cost = costs.column_cost + costs.neuron_cost * self.config.min_neurons_per_column
            if self.budget.can_afford(cost):
                operations.append({
                    'type': GrowthEvent.ADD_COLUMN,
                    'neuron_count': self.config.min_neurons_per_column,
                    'cost': cost
                })
        
        return operations
    
    def plan_pruning(self, activation_counts: Dict[int, int],
                     weight_magnitudes: Dict[Tuple[int, int], float]) -> List[Dict[str, Any]]:
        """
        Plan pruning operations.
        
        Args:
            activation_counts: Dict mapping column_id to activation count
            weight_magnitudes: Dict mapping (from, to) to weight magnitude
        
        Returns:
            List of pruning operations to perform
        """
        should_prune, reason = self.should_prune()
        if not should_prune:
            return []
        
        operations = []
        
        # Find columns to prune (low activation)
        columns_to_prune = []
        for col_id, count in activation_counts.items():
            if count < self.config.activation_prune_threshold:
                columns_to_prune.append(col_id)
        
        # Limit pruning
        columns_to_prune = columns_to_prune[:self.config.prune_batch_size]
        
        for col_id in columns_to_prune:
            if self.num_columns - len(operations) > self.config.min_columns:
                operations.append({
                    'type': GrowthEvent.PRUNE_COLUMN,
                    'column_id': col_id,
                    'refund': self.config.costs.column_cost * 0.5  # Partial refund
                })
        
        # Find connections to prune (weak weights)
        connections_to_prune = []
        for (from_id, to_id), magnitude in weight_magnitudes.items():
            if magnitude < self.config.weight_prune_threshold:
                connections_to_prune.append((from_id, to_id))
        
        # Limit connection pruning
        for conn in connections_to_prune[:self.config.prune_batch_size * 10]:
            operations.append({
                'type': GrowthEvent.PRUNE_CONNECTION,
                'from_id': conn[0],
                'to_id': conn[1],
                'refund': self.config.costs.connection_cost * 0.1
            })
        
        return operations
    
    def execute_growth(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute growth operations and update tracking.
        
        Args:
            operations: List of growth operations
        
        Returns:
            List of successfully executed operations
        """
        executed = []
        
        for op in operations:
            if op['type'] == GrowthEvent.ADD_NEURON:
                if self.budget.spend(op['cost'], f"add_neurons_{op['column_id']}"):
                    self.neurons_per_column[op['column_id']] = \
                        self.neurons_per_column.get(op['column_id'], 0) + op['count']
                    executed.append(op)
                    self._log_event(op)
            
            elif op['type'] == GrowthEvent.ADD_COLUMN:
                if self.budget.spend(op['cost'], "add_column"):
                    new_col_id = max(self.neurons_per_column.keys(), default=-1) + 1
                    self.neurons_per_column[new_col_id] = op['neuron_count']
                    self.num_columns += 1
                    op['column_id'] = new_col_id
                    executed.append(op)
                    self._log_event(op)
        
        return executed
    
    def execute_pruning(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute pruning operations and update tracking.
        
        Args:
            operations: List of pruning operations
        
        Returns:
            List of successfully executed operations
        """
        executed = []
        
        for op in operations:
            if op['type'] == GrowthEvent.PRUNE_COLUMN:
                if op['column_id'] in self.neurons_per_column:
                    del self.neurons_per_column[op['column_id']]
                    self.num_columns -= 1
                    self.budget.current_budget += op.get('refund', 0)
                    executed.append(op)
                    self._log_event(op)
            
            elif op['type'] == GrowthEvent.PRUNE_CONNECTION:
                self.total_connections -= 1
                self.budget.current_budget += op.get('refund', 0)
                executed.append(op)
                self._log_event(op)
        
        return executed
    
    def _log_event(self, operation: Dict[str, Any]):
        """Log a growth/pruning event."""
        event = {
            'tick': self.current_tick,
            'type': operation['type'].value if isinstance(operation['type'], GrowthEvent) else operation['type'],
            **{k: v for k, v in operation.items() if k != 'type'}
        }
        self.events.append(event)
    
    def tick(self):
        """Advance one tick (regenerate budget, update tick counter)."""
        self.current_tick += 1
        self.budget.regenerate(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get growth controller statistics."""
        return {
            'current_tick': self.current_tick,
            'num_columns': self.num_columns,
            'total_neurons': sum(self.neurons_per_column.values()),
            'total_connections': self.total_connections,
            'current_cost': self.compute_current_cost(),
            'budget': self.budget.get_stats(),
            'metrics': self.metrics.get_summary(),
            'total_events': len(self.events)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"GrowthController(columns={stats['num_columns']}, "
                f"neurons={stats['total_neurons']}, "
                f"budget={stats['budget']['current_budget']:.1f})")


if __name__ == "__main__":
    print("=== Growth Controller Tests ===\n")
    
    # Create controller
    config = GrowthConfig(
        initial_budget=500.0,
        max_budget=1000.0
    )
    
    controller = GrowthController(config)
    
    # Initialize with some columns
    controller.set_network_size(
        num_columns=8,
        neurons_per_column={i: 16 for i in range(8)},
        total_connections=1000
    )
    
    print(f"Created: {controller}")
    print(f"Initial stats: {controller.get_stats()}\n")
    
    # Simulate high utilization scenario
    print("Simulating high utilization...")
    for i in range(50):
        controller.metrics.record_utilization(0.95)
        controller.metrics.record_novelty(0.3)
        controller.metrics.record_task_result(success=True, reward=1.0)
        controller.tick()
    
    # Check growth decision
    should_grow, reason = controller.should_grow()
    print(f"Should grow: {should_grow}, reason: {reason}")
    
    # Plan and execute growth
    growth_ops = controller.plan_growth()
    print(f"Planned growth operations: {len(growth_ops)}")
    
    if growth_ops:
        executed = controller.execute_growth(growth_ops)
        print(f"Executed: {len(executed)} operations")
    
    print(f"\nAfter growth: {controller}")
    print(f"Stats: {controller.get_stats()}\n")
    
    # Simulate low utilization scenario
    print("Simulating low utilization...")
    for i in range(50):
        controller.metrics.record_utilization(0.2)
        controller.metrics.record_novelty(0.1)
        controller.metrics.record_task_result(success=True, reward=1.0)
        controller.tick()
    
    # Check pruning decision
    should_prune, reason = controller.should_prune()
    print(f"Should prune: {should_prune}, reason: {reason}")
    
    # Plan and execute pruning
    activation_counts = {i: 5 if i < 2 else 100 for i in range(8)}
    weight_magnitudes = {}
    
    prune_ops = controller.plan_pruning(activation_counts, weight_magnitudes)
    print(f"Planned pruning operations: {len(prune_ops)}")
    
    if prune_ops:
        executed = controller.execute_pruning(prune_ops)
        print(f"Executed: {len(executed)} operations")
    
    print(f"\nAfter pruning: {controller}")
    print(f"Final stats: {controller.get_stats()}")
