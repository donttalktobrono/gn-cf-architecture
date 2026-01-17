# Smart Manus AGI Neural Network Architecture
## Balanced Ternary / Base-5 Self-Organizing Cognitive System
### Version 1.0 - January 2026

---

## Executive Summary

This document specifies a novel neural network architecture that combines balanced ternary computation with self-organizing modular structure. The system is designed to exhibit emergent intelligence through structural adaptation rather than simply scaling parameters. Key innovations include a dual-role training paradigm (Experiencer/Saboteur sharing the same network), affinity-based liquid clustering for skill specialization, and Hebbian coupling for natural module formation.

---

## 1. Foundational Principles

### 1.1 Why Balanced Ternary?

Traditional neural networks use floating-point weights, which are computationally expensive and biologically implausible. Balanced ternary offers several advantages:

| Property | Binary | Balanced Ternary | Benefit |
|----------|--------|------------------|---------|
| Weight Values | {0, 1} or {-1, +1} | {-1, 0, +1} | Natural inhibition/excitation |
| Multiplication | Requires compute | Trivial (sign flip or zero) | 10-100x faster inference |
| Sparsity | Separate mechanism | Built-in (0 = no connection) | Natural pruning |
| Sign Handling | Separate bit | Implicit in representation | Simpler hardware |
| Biological Plausibility | Low | High | Matches synaptic behavior |

The balanced ternary system uses digits {T, 0, 1} where T represents -1. Any integer can be represented without a separate sign, and arithmetic operations are straightforward.

### 1.2 Base-5 (Quinary) Activations

While weights use ternary values, neuron activations use balanced quinary {-2, -1, 0, +1, +2} to provide greater expressiveness:

| Activation Value | Semantic Meaning | Biological Analog |
|-----------------|------------------|-------------------|
| -2 | Strong Inhibition | Hyperpolarization |
| -1 | Weak Inhibition | Sub-threshold inhibition |
| 0 | Inactive/Neutral | Resting potential |
| +1 | Weak Excitation | Sub-threshold excitation |
| +2 | Strong Excitation | Action potential |

This creates a 5-level activation system that captures nuanced neural states while remaining computationally efficient.

### 1.3 Mixed-Radix Computation

The architecture uses mixed-radix arithmetic where different components operate in different bases:

```
Weights:      Base-3 (Balanced Ternary)  → {-1, 0, +1}
Activations:  Base-5 (Balanced Quinary)  → {-2, -1, 0, +1, +2}
Routing:      Base-3 (Ternary Logic)     → {Inhibit, Neutral, Excite}
Affinity:     Base-5 (Similarity Score)  → {Repel, Weak-, Neutral, Weak+, Attract}
```

---

## 2. Core Architecture Components

### 2.1 Ternary Neuron Model

Each neuron in the network follows this computational model:

```python
class TernaryNeuron:
    """
    A single neuron with ternary weights and quinary activation.
    
    Computation:
    1. Weighted sum: z = Σ(w_i * x_i) where w_i ∈ {-1, 0, +1}
    2. Bias addition: z' = z + b where b ∈ {-2, -1, 0, +1, +2}
    3. Quantized activation: a = quinary_activation(z')
    """
    
    def forward(self, inputs):
        # Ternary dot product (no multiplication needed)
        weighted_sum = 0
        for w, x in zip(self.weights, inputs):
            if w == 1:
                weighted_sum += x
            elif w == -1:
                weighted_sum -= x
            # w == 0: no contribution
        
        # Add bias and apply quinary activation
        z = weighted_sum + self.bias
        return self.quinary_activate(z)
    
    def quinary_activate(self, z):
        # Map continuous value to {-2, -1, 0, +1, +2}
        if z <= -1.5:
            return -2
        elif z <= -0.5:
            return -1
        elif z <= 0.5:
            return 0
        elif z <= 1.5:
            return 1
        else:
            return 2
```

### 2.2 Microcolumn Structure (from GN-CF)

Neurons are organized into microcolumns, which are the basic computational units:

```
Microcolumn (Expert Unit)
├── Input Layer (receives broadcast from Workspace)
├── Hidden Layers (ternary weights, quinary activations)
├── Output Layer (produces contribution to global state)
├── Affinity Vector (soft assignment to Liquids)
├── Local State (persistent memory within column)
└── Position (2D coordinates on the Sheet)
```

Each microcolumn has:
- **Index**: Unique identifier (0 to N-1 where N = S×S for an S×S sheet)
- **Position**: (x, y) coordinates for spatial relationships
- **Affinity**: K-dimensional vector (simplex) over Liquids
- **Weights**: Ternary connection weights
- **State**: Local persistent state vector

### 2.3 The Sheet (GyrusNet)

Microcolumns are arranged in a 2D sheet, inspired by cortical organization:

```
┌─────────────────────────────────────────┐
│  Sheet (S × S Microcolumns)             │
│                                         │
│  ○──○──○──○──○──○──○──○  ← Row 0       │
│  │╲ │╲ │╲ │╲ │╲ │╲ │╲ │                │
│  ○──○──○──○──○──○──○──○  ← Row 1       │
│  │╲ │╲ │╲ │╲ │╲ │╲ │╲ │                │
│  ○──○──○──○──○──○──○──○  ← Row 2       │
│  ...                                    │
│                                         │
│  Legend:                                │
│  ○ = Microcolumn                        │
│  ─ = Horizontal connection              │
│  │ = Vertical connection                │
│  ╲ = Diagonal connection                │
└─────────────────────────────────────────┘
```

**Connectivity Pattern (LE/LI - Local Excitation / Lateral Inhibition)**:

```python
def compute_connection_weight(pos_i, pos_j, sigma_e=1.0, sigma_i=3.0):
    """
    Mexican hat connectivity: local excitation, lateral inhibition.
    Returns ternary weight: +1 (excite), 0 (none), -1 (inhibit)
    """
    distance = euclidean_distance(pos_i, pos_j)
    
    # Difference of Gaussians (Mexican hat)
    excitation = exp(-distance**2 / (2 * sigma_e**2))
    inhibition = exp(-distance**2 / (2 * sigma_i**2))
    
    raw_weight = excitation - inhibition
    
    # Quantize to ternary
    if raw_weight > 0.3:
        return +1  # Local excitation
    elif raw_weight < -0.1:
        return -1  # Lateral inhibition
    else:
        return 0   # No connection
```

---

## 3. Affinity Liquids System

### 3.1 Concept

Liquids are dynamic skill primitives that represent clusters of related capabilities. Unlike fixed expert modules, Liquids can spawn, merge, and evolve based on task demands.

```
Liquid Properties:
├── Prototype Vector: P_k ∈ ℝ^d (centroid of skill cluster)
├── Permissions: Which tools/actions this liquid can access
├── Budgets: Resource limits (compute, memory, tool calls)
├── Member Columns: Microcolumns with high affinity to this liquid
└── Activation History: When and how often this liquid is used
```

### 3.2 Affinity Computation

Each microcolumn maintains a soft assignment to all Liquids:

```python
def compute_affinity(column_embedding, liquid_prototypes, temperature=1.0):
    """
    Compute affinity scores (simplex over K liquids).
    Uses ternary-aware similarity.
    """
    K = len(liquid_prototypes)
    scores = []
    
    for k in range(K):
        # Ternary dot product similarity
        sim = ternary_similarity(column_embedding, liquid_prototypes[k])
        scores.append(sim / temperature)
    
    # Softmax to get simplex (sums to 1)
    affinity = softmax(scores)
    
    # Quantize to balanced quinary for efficiency
    quantized = [quinary_quantize(a * 4 - 2) for a in affinity]
    
    return quantized  # Each value in {-2, -1, 0, +1, +2}
```

### 3.3 Liquid Lifecycle

Liquids evolve through Structure Ticks (periodic consolidation events):

```
Structure Tick Process:
1. Collect novelty events from buffer
2. Cluster events by embedding similarity
3. SPAWN: Create new liquid if cluster is:
   - Large enough (≥ M_min events)
   - Persistent (≥ T_stable ticks)
   - Distinct (similarity to existing < τ_spawn)
4. MERGE: Combine liquids if:
   - Cosine similarity > τ_merge
   - Context overlap (JSD) < τ_merge_jsd
5. PRUNE: Remove liquids with low utilization
6. CONSOLIDATE: Replay + distill + retention check
7. ROLLBACK: Undo structural changes if retention fails
```

---

## 4. Dual-Role Training: Experiencer vs Saboteur

### 4.1 Core Concept

The same neural network operates in two modes, sharing all weights, liquids, and memory:

```
┌─────────────────────────────────────────────────────────┐
│                    SHARED BRAIN                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Workspace (CortexFormer)                       │   │
│  │  Sheet (GyrusNet with Microcolumns)             │   │
│  │  Liquids (Skill Primitives)                     │   │
│  │  Memory (Episodic + Semantic)                   │   │
│  │  Tools (Code, Search, Math, etc.)               │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│              ┌──────────┴──────────┐                   │
│              │                     │                   │
│              ▼                     ▼                   │
│     ┌─────────────┐       ┌─────────────┐             │
│     │ EXPERIENCER │       │  SABOTEUR   │             │
│     │   Mode +1   │       │   Mode -1   │             │
│     │             │       │             │             │
│     │ Goal: Solve │       │ Goal: Break │             │
│     │ the task    │       │ the solver  │             │
│     └─────────────┘       └─────────────┘             │
│                                                        │
│  Mode Signal: m ∈ {-1, 0, +1} (Ternary!)              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Mode-Conditioned Computation

The mode signal modulates network behavior:

```python
class DualRoleNetwork:
    def forward(self, input, mode):
        """
        mode: +1 (Experiencer), -1 (Saboteur), 0 (Evaluate)
        """
        # Embed mode as ternary signal
        mode_embedding = self.mode_encoder(mode)  # Ternary vector
        
        # Workspace processes input with mode context
        workspace_state = self.workspace(input, mode_embedding)
        
        # Router selects experts (mode affects routing)
        selected, weights = self.router(workspace_state, mode)
        
        # Experts compute (same weights, different objectives)
        outputs = []
        for idx, w in zip(selected, weights):
            expert_out = self.sheet[idx].forward(workspace_state)
            outputs.append(w * expert_out)
        
        # Aggregate
        result = sum(outputs)
        
        # Mode-specific head
        if mode == 1:  # Experiencer
            return self.solve_head(result)
        elif mode == -1:  # Saboteur
            return self.challenge_head(result)
        else:  # Evaluate
            return self.evaluate_head(result)
```

### 4.3 Training Dynamics

The dual-role training creates a natural adversarial curriculum:

```
Training Loop:
┌─────────────────────────────────────────────────────────┐
│ 1. SABOTEUR PHASE (mode = -1)                          │
│    - Generate challenge/modification to current task    │
│    - Constraint: Challenge must be theoretically        │
│      solvable (saboteur could solve it in mode +1)     │
│                                                         │
│ 2. EXPERIENCER PHASE (mode = +1)                       │
│    - Attempt to solve the saboteur's challenge         │
│    - Record success/failure and difficulty             │
│                                                         │
│ 3. EVALUATION PHASE (mode = 0)                         │
│    - Assess: Was challenge appropriate?                │
│    - Compute regret: |expected_difficulty - actual|    │
│                                                         │
│ 4. UPDATE PHASE                                        │
│    - Update shared weights to:                         │
│      a) Improve experiencer's solving ability          │
│      b) Improve saboteur's challenge calibration       │
│      c) Minimize regret (challenge appropriateness)    │
└─────────────────────────────────────────────────────────┘
```

### 4.4 Ternary Reward Signal

```python
def compute_dual_role_reward(experiencer_success, challenge_difficulty, 
                             experiencer_capability):
    """
    Returns ternary reward for the dual-role system.
    """
    if experiencer_success:
        if challenge_difficulty > experiencer_capability + 0.1:
            return +1  # Good: Solved a hard challenge (growth)
        else:
            return 0   # Neutral: Solved easy challenge (no growth)
    else:
        if challenge_difficulty < experiencer_capability + 0.3:
            return -1  # Bad: Failed achievable challenge (regression)
        else:
            return 0   # Neutral: Failed very hard challenge (expected)
```

---

## 5. Hebbian Coupling and Module Formation

### 5.1 Ternary Hebbian Rule

The classic Hebbian principle "cells that fire together wire together" is implemented with ternary updates:

```python
def hebbian_update(pre_activation, post_activation, current_weight, 
                   learning_rate=0.1):
    """
    Ternary Hebbian learning rule.
    
    pre_activation, post_activation: Quinary values {-2, -1, 0, +1, +2}
    current_weight: Ternary value {-1, 0, +1}
    
    Returns: New ternary weight
    """
    # Compute correlation
    correlation = pre_activation * post_activation
    
    # Determine update direction
    if correlation > 2:  # Both strongly active (same sign)
        delta = +1  # Strengthen
    elif correlation < -2:  # Opposite activations
        delta = -1  # Weaken
    else:
        delta = 0  # No change
    
    # Apply update with learning rate (stochastic)
    if random() < learning_rate:
        new_weight = current_weight + delta
        # Clamp to ternary range
        return max(-1, min(+1, new_weight))
    else:
        return current_weight
```

### 5.2 Module Formation Through LE/LI + Hebbian

The combination of Local Excitation/Lateral Inhibition with Hebbian learning naturally creates modular structure:

```
Initial State:          After Training:
┌───────────────┐      ┌───────────────┐
│ ○ ○ ○ ○ ○ ○ ○ │      │ ●─●─● ○ ◆─◆─◆ │
│ ○ ○ ○ ○ ○ ○ ○ │      │ │╲│╲│   │╲│╲│ │
│ ○ ○ ○ ○ ○ ○ ○ │  →   │ ●─●─● ○ ◆─◆─◆ │
│ ○ ○ ○ ○ ○ ○ ○ │      │       ○       │
│ ○ ○ ○ ○ ○ ○ ○ │      │ ▲─▲─▲ ○ ■─■─■ │
└───────────────┘      └───────────────┘

Legend:
○ = Inactive/Neutral
●, ◆, ▲, ■ = Different modules (Liquids)
─, │, ╲ = Strong connections within module
(spaces) = Weak/inhibitory connections between modules
```

### 5.3 Characteristic Wavelength

The LE/LI connectivity creates a natural spatial scale for modules:

```python
def compute_characteristic_wavelength(sigma_e, sigma_i):
    """
    The characteristic wavelength Λ determines module size.
    Derived from the Mexican hat connectivity profile.
    """
    # Approximate formula from Turing pattern theory
    lambda_char = 2 * pi * sqrt(2) * sqrt(sigma_e * sigma_i)
    return lambda_char

# Example: With sigma_e=1.0, sigma_i=3.0
# Λ ≈ 7.7 units → modules of ~8x8 microcolumns
```

---

## 6. Dynamic Growth and Pruning

### 6.1 Growth Triggers

The network can grow new neurons/microcolumns when:

```python
def should_grow(metrics, thresholds):
    """
    Determine if network should add capacity.
    """
    triggers = [
        metrics['utilization'] > thresholds['high_utilization'],
        metrics['surprise_score'] > thresholds['novelty'],
        metrics['task_failure_rate'] > thresholds['failure'],
        metrics['liquid_saturation'] > thresholds['saturation']
    ]
    return any(triggers)
```

### 6.2 Pruning Triggers

Neurons/connections are pruned when:

```python
def should_prune(neuron, metrics, thresholds):
    """
    Determine if a neuron should be pruned.
    """
    triggers = [
        neuron.activation_count < thresholds['min_activations'],
        neuron.weight_magnitude < thresholds['min_weight'],
        metrics['redundancy_score'] > thresholds['redundancy'],
        metrics['resource_pressure'] > thresholds['pressure']
    ]
    return any(triggers)
```

### 6.3 Resource Cost Model

Growth has a cost to prevent unbounded expansion:

```python
class ResourceCostModel:
    def __init__(self):
        self.neuron_cost = 1.0      # Cost per neuron
        self.connection_cost = 0.1  # Cost per connection
        self.liquid_cost = 10.0     # Cost per liquid
        self.memory_cost = 0.01     # Cost per memory slot
    
    def compute_total_cost(self, network):
        cost = 0
        cost += self.neuron_cost * network.num_neurons
        cost += self.connection_cost * network.num_connections
        cost += self.liquid_cost * network.num_liquids
        cost += self.memory_cost * network.memory_usage
        return cost
    
    def growth_penalty(self, proposed_growth, budget):
        """
        Penalty for growth that exceeds budget.
        """
        new_cost = self.compute_total_cost_after_growth(proposed_growth)
        if new_cost > budget:
            return (new_cost - budget) ** 2
        return 0
```

---

## 7. Tool Integration

### 7.1 Tools as First-Class Actions

Tools are integrated into the action space, not bolted on:

```python
TOOL_REGISTRY = {
    'code_exec': {
        'description': 'Execute Python code',
        'input_type': 'string',
        'output_type': 'string',
        'permission_level': 2,
        'cost': 5.0
    },
    'search': {
        'description': 'Search the web',
        'input_type': 'string',
        'output_type': 'list[string]',
        'permission_level': 1,
        'cost': 1.0
    },
    'math': {
        'description': 'Compute mathematical expression',
        'input_type': 'string',
        'output_type': 'number',
        'permission_level': 0,
        'cost': 0.1
    },
    'memory_write': {
        'description': 'Write to long-term memory',
        'input_type': 'tuple[key, value]',
        'output_type': 'bool',
        'permission_level': 2,
        'cost': 2.0
    },
    'memory_read': {
        'description': 'Read from long-term memory',
        'input_type': 'key',
        'output_type': 'value',
        'permission_level': 1,
        'cost': 0.5
    }
}
```

### 7.2 Permission System

Each Liquid has permissions for which tools it can access:

```python
class LiquidPermissions:
    def __init__(self, liquid_id):
        self.liquid_id = liquid_id
        self.allowed_tools = set()
        self.budgets = {}
        self.usage = {}
    
    def can_use_tool(self, tool_name):
        if tool_name not in self.allowed_tools:
            return False
        if self.usage.get(tool_name, 0) >= self.budgets.get(tool_name, 0):
            return False
        return True
    
    def use_tool(self, tool_name):
        if self.can_use_tool(tool_name):
            self.usage[tool_name] = self.usage.get(tool_name, 0) + 1
            return True
        return False
```

---

## 8. Memory Architecture

### 8.1 Tiered Memory System

```
Memory Tiers:
┌─────────────────────────────────────────────────────────┐
│ WORKSPACE MEMORY (Immediate Context)                    │
│ - Current task state                                    │
│ - Active tokens/embeddings                              │
│ - Capacity: ~4K tokens                                  │
│ - Persistence: Single episode                           │
├─────────────────────────────────────────────────────────┤
│ EPISODIC MEMORY (Recent Experiences)                    │
│ - Recent episodes and outcomes                          │
│ - Indexed by context similarity                         │
│ - Capacity: ~1000 episodes                              │
│ - Persistence: Decays over time                         │
├─────────────────────────────────────────────────────────┤
│ SEMANTIC MEMORY (Learned Knowledge)                     │
│ - Consolidated facts and skills                         │
│ - Indexed by concept embeddings                         │
│ - Capacity: Grows with learning                         │
│ - Persistence: Long-term                                │
├─────────────────────────────────────────────────────────┤
│ AUDIT LOG (Immutable Record)                            │
│ - All memory writes                                     │
│ - Tool usage                                            │
│ - Decision traces                                       │
│ - Persistence: Permanent                                │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Memory Write Governance

All memory writes are governed by the permission system:

```python
def write_to_memory(tier, key, value, liquid_id, justification):
    """
    Governed memory write with audit trail.
    """
    # Check permission
    if not liquids[liquid_id].can_use_tool('memory_write'):
        raise PermissionError(f"Liquid {liquid_id} cannot write to memory")
    
    # Check budget
    if memory_budget_exceeded(liquid_id):
        raise BudgetError(f"Liquid {liquid_id} exceeded memory budget")
    
    # Compute predicted utility
    utility = predict_write_utility(key, value, justification)
    if utility < MIN_WRITE_UTILITY:
        raise LowUtilityError(f"Write utility {utility} below threshold")
    
    # Perform write
    memory[tier][key] = value
    
    # Audit
    audit_log.append({
        'action': 'memory_write',
        'tier': tier,
        'key': key,
        'value_hash': hash(value),
        'liquid_id': liquid_id,
        'justification': justification,
        'utility': utility,
        'timestamp': now()
    })
    
    return True
```

---

## 9. Complete System Architecture

### 9.1 Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │           INPUT                     │
                    │  (Task, Environment State, Mode)    │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         WORKSPACE (CortexFormer)                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Token Embedding → Transformer Layers → Hidden State H_t       │ │
│  │                                                                 │ │
│  │  Mode Embedding (Ternary: -1/0/+1) injected at each layer      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│                    Router Query q_t = pool(H_t)                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         ROUTER (Thalamic)                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  For each microcolumn i:                                        │ │
│  │    key_i = W_k · [E_i, A_i, pos_i, perm_i]                     │ │
│  │    logit_i = q_t · key_i - λ_wire·wire(i) - λ_cost·cost(i)     │ │
│  │                                                                 │ │
│  │  Select S_t = TopK(logits, k)                                  │ │
│  │  Weights w = softmax(logits[S_t])                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         SHEET (GyrusNet)                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  2D Grid of Microcolumns (S × S)                               │ │
│  │                                                                 │ │
│  │  ┌─────┐  ┌─────┐  ┌─────┐       ┌─────┐                       │ │
│  │  │ MC₀ │──│ MC₁ │──│ MC₂ │─ ... ─│MCₙ₋₁│                       │ │
│  │  └──┬──┘  └──┬──┘  └──┬──┘       └──┬──┘                       │ │
│  │     │        │        │             │                           │ │
│  │  ┌──▼──┐  ┌──▼──┐  ┌──▼──┐       ┌──▼──┐                       │ │
│  │  │Liquid│  │Liquid│  │Liquid│     │Liquid│  (Affinity)         │ │
│  │  │  A  │  │  B  │  │  A  │       │  C  │                       │ │
│  │  └─────┘  └─────┘  └─────┘       └─────┘                       │ │
│  │                                                                 │ │
│  │  LE/LI Connectivity: Local +1, Lateral -1, Distant 0           │ │
│  │  Hebbian Updates: Fire together → wire together                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                   Selected columns compute                           │
│                              │                                       │
│                              ▼                                       │
│                    Weighted sum of outputs                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      MODE-SPECIFIC HEADS                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │  EXPERIENCER   │  │   SABOTEUR     │  │   EVALUATOR    │         │
│  │   (mode=+1)    │  │   (mode=-1)    │  │   (mode=0)     │         │
│  │                │  │                │  │                │         │
│  │ Output: Action │  │ Output: Chall. │  │ Output: Score  │         │
│  │ to solve task  │  │ modification   │  │ assessment     │         │
│  └────────────────┘  └────────────────┘  └────────────────┘         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         TOOL INTERFACE                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  If action is tool call:                                        │ │
│  │    1. Check permissions (Liquid → Tool mapping)                 │ │
│  │    2. Check budget                                              │ │
│  │    3. Execute tool                                              │ │
│  │    4. Return result as new tokens                               │ │
│  │    5. Audit log                                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │           OUTPUT                    │
                    │  (Action, Tool Call, or Response)   │
                    └─────────────────────────────────────┘
```

### 9.2 Training Loop

```python
def training_step(env, network, episode_buffer):
    """
    Single training step with dual-role dynamics.
    """
    # Get current task from environment
    task = env.get_task()
    
    # PHASE 1: Saboteur generates challenge
    network.set_mode(-1)  # Saboteur mode
    challenge = network.forward(task)
    modified_task = env.apply_challenge(task, challenge)
    
    # PHASE 2: Experiencer attempts challenge
    network.set_mode(+1)  # Experiencer mode
    done = False
    total_reward = 0
    
    while not done:
        action = network.forward(env.get_state())
        
        if is_tool_call(action):
            result = execute_tool(action, network.current_liquid)
            env.inject_observation(result)
        else:
            next_state, reward, done = env.step(action)
            total_reward += reward
    
    # PHASE 3: Evaluate and compute losses
    network.set_mode(0)  # Evaluation mode
    difficulty_estimate = network.forward(modified_task)
    
    # Compute regret
    expected_success = sigmoid(network.capability - difficulty_estimate)
    actual_success = total_reward > 0
    regret = abs(expected_success - actual_success)
    
    # Compute losses
    experiencer_loss = -total_reward  # Maximize reward
    saboteur_loss = -regret  # Maximize appropriate challenge
    hebbian_loss = compute_hebbian_loss(network)  # Encourage clustering
    
    # Update
    total_loss = experiencer_loss + saboteur_loss + hebbian_loss
    network.backward(total_loss)
    network.update_weights()
    
    # Periodic structure tick
    if should_structure_tick():
        network.structure_tick()
    
    return total_reward, regret
```

---

## 10. Evaluation Gates

Following the GN-CF framework, we define gates that must pass before scaling:

| Gate ID | Metrics | Thresholds |
|---------|---------|------------|
| router_health | top1_share, dead_expert_frac, selection_entropy | <0.2, <0.05, >1.5 |
| liquid_lifecycle | k_growth_rate, spawn_reuse_rate, redundant_pair_frac | <0.001, >0.6, <0.1 |
| retention | max_regression_frac, merge_event_regression | <0.1, <0.05 |
| tool_roi_and_safety | waste_rate, permission_violations, median_roi | <0.3, 0, >0.0 |
| memory_integrity | write_precision, poison_recovery, write_budget | >0.5, <3, <3 |
| dual_role_balance | experiencer_win_rate, saboteur_calibration | 0.4-0.6, <0.2 |
| hebbian_modularity | intra_module_strength, inter_module_inhibition | >0.7, <-0.3 |

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Implement ternary neuron and quinary activation
- Build basic microcolumn structure
- Create 2D sheet with LE/LI connectivity

### Phase 2: Liquids (Weeks 3-4)
- Implement affinity computation
- Build liquid lifecycle (spawn/merge/prune)
- Add structure tick mechanism

### Phase 3: Dual-Role (Weeks 5-6)
- Implement mode-conditioned computation
- Build experiencer and saboteur heads
- Create training loop with regret minimization

### Phase 4: Hebbian (Weeks 7-8)
- Implement ternary Hebbian updates
- Add module formation metrics
- Tune LE/LI parameters for characteristic wavelength

### Phase 5: Tools & Memory (Weeks 9-10)
- Integrate tool registry
- Build permission system
- Implement tiered memory with governance

### Phase 6: Testing (Weeks 11-12)
- Create Flappy Bird environment
- Run experiments
- Validate emergent properties

---

## 12. Success Criteria

The architecture will be considered successful if it demonstrates:

1. **Self-Organization**: Different agents evolve different neuron counts
2. **Affinity Clustering**: Clear regions form on the sheet
3. **Hebbian Modules**: Links concentrate within regions
4. **Adaptive Growth**: Harder tasks push growth; easy tasks favor efficiency
5. **Adversarial Co-Evolution**: Experiencer and saboteur improve together
6. **Tool Specialization**: Different liquids specialize in different tools
7. **Memory Governance**: No unauthorized writes; high precision

---

*This architecture document serves as the blueprint for implementation. Each component has been designed to work together, creating a system where intelligence emerges from structure rather than just scale.*
