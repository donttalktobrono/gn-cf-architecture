# Balanced Ternary Neural Network Research Notes

## Balanced Ternary Fundamentals

### Definition
- **Digits**: {-1, 0, 1} (often written as {T, 0, 1} where T = -1)
- **Base**: 3 (ternary)
- **Key Property**: No separate minus sign needed - the leading non-zero digit indicates sign

### Value Calculation
For a string d_n...d_0:
```
v(d_n...d_0) = Σ f(d_i) * 3^i
```
where f(T) = -1, f(0) = 0, f(1) = 1

### Arithmetic Tables

**Addition Table:**
```
+   | T   0   1
----|----------
T   | T1  T   0
0   | T   0   1
1   | 0   1   1T
```

**Subtraction Table:**
```
-   | T   0   1
----|----------
T   | 0   T   T1
0   | 1   0   T
1   | 1T  1   0
```

**Multiplication Table:**
```
×   | T   0   1
----|----------
T   | 1   0   T
0   | 0   0   0
1   | T   0   1
```

**Division Table:**
```
÷   | T   1
----|------
T   | 1   T
0   | 0   0
1   | T   1
```

### Key Advantages for Neural Networks

1. **Natural Sign Representation**: Negative numbers don't need special handling
2. **Symmetric Around Zero**: Values {-1, 0, 1} are balanced
3. **Efficient Multiplication**: Multiplication by -1, 0, or 1 is trivial
4. **Sparsity**: Zero values mean no computation needed
5. **No Overflow Concerns**: Bounded weights naturally
6. **Rounding Simplicity**: Round to nearest of {-1, 0, 1}

### Conversion Examples
- 8dec = 10T (= 9 - 1)
- -9dec = T00 (= -9)
- 29dec = 101T (= 27 + 3 - 1)

### Multi-trit Operations
- Add/subtract trit by trit with carry
- Carry propagation similar to binary but with three states

## Ternary Weight Networks (TWN) Research

### Key Papers
1. **Ternary Weight Networks (Li et al., 2016)** - 1468 citations
   - Weights quantized to {-1, 0, +1}
   - Stronger expressive ability than binary networks
   - Multiplication-free inference possible

2. **FATNN: Fast and Accurate Ternary Neural Networks (Chen et al., 2021)**
   - Focus on model quantization
   - Reduces complexity with low-precision weights/activations

3. **Ternary Neural Networks for Resource-Efficient AI (Alemdar et al., 2017)** - 340 citations
   - Suitable for edge devices, smartphones, wearables
   - Significant memory/computation reduction

### Quantization Approaches
- Map real-valued parameters to {-α, 0, +α}
- α is a learned or computed scaling factor
- Threshold-based assignment to ternary values

## Base-5 (Quinary) Integration Concept

### Why Combine Ternary with Base-5?
- Base-5 provides 5 states: {-2, -1, 0, 1, 2} (balanced quinary)
- Or standard quinary: {0, 1, 2, 3, 4}
- More expressive than ternary while still efficient

### Mixed-Radix Approach
- Different layers/components can use different bases
- Ternary for weights (efficiency)
- Quinary for activations (expressiveness)
- Allows fine-grained control over precision/efficiency tradeoff

### RadiX-Net Concept (Kepner & Robinett, 2019)
- Structured sparse matrices using mixed-radix topologies
- Feedforward neural net topology based on number theory
- Can create efficient sparse connectivity patterns

## Application to AGI Architecture

### Proposed Hybrid System
1. **Ternary Weights**: {-1, 0, 1} for synaptic connections
   - Sparse (0 = no connection)
   - Excitatory (1) or Inhibitory (-1)
   - Biologically plausible

2. **Quinary Activations**: {-2, -1, 0, 1, 2} for neuron states
   - Strong inhibition (-2)
   - Weak inhibition (-1)
   - Inactive (0)
   - Weak excitation (1)
   - Strong excitation (2)

3. **Affinity Liquids**: Use ternary similarity scores
   - -1: Repulsion (different skill types)
   - 0: Neutral
   - 1: Attraction (similar skills cluster)

4. **Hebbian Updates**: Ternary delta rules
   - Strengthen (+1), Weaken (-1), or No change (0)

### Computational Benefits
- Integer-only arithmetic possible
- No floating point needed
- Hardware-efficient (can use simple logic gates)
- Natural sparsity encourages modular structure


## Self-Organization of Modular Neural Networks

### Key Findings from Nature Communications (Mulholland et al., 2024)

#### Local Excitation / Lateral Inhibition (LE/LI) Mechanism
- **Core Principle**: Modular patterns emerge from networks with:
  - Local excitation (short-range positive connections)
  - Lateral inhibition (longer-range negative connections)
- Based on Alan Turing's morphogenesis framework (1952)
- Creates characteristic spatial wavelength (Λ)

#### Key Properties
1. **Characteristic Wavelength**: Networks naturally organize activity at a specific spatial scale
2. **Selective Amplification**: The network amplifies patterns matching its characteristic wavelength
3. **Stability + Flexibility**: Robust modular organization with flexible absolute positions
4. **Self-Organization**: No structured inputs needed - emerges from recurrent interactions

#### Implications for AGI Architecture
- **Affinity Liquids**: Can self-organize using LE/LI principles
- **Module Formation**: Local excitation within skill clusters, lateral inhibition between different skills
- **Characteristic Scale**: Each "lobe" has a natural size determined by connectivity
- **Dynamic Emergence**: Modules form through activity, not hardcoding

### Hebbian Learning for Clustering

#### Core Principle: "Cells that fire together wire together"
- Synaptic connections strengthen when pre- and post-synaptic neurons activate together
- Creates natural clustering of co-activated features

#### Statistical Basis (Nonlinear Hebbian Learning)
- Can perform unsupervised clustering in input space
- Decorrelated Hebbian Learning (DHL) avoids winner-take-all
- Learns distributed representations naturally

#### Application to Affinity Liquids
1. **Intra-module strengthening**: Neurons in same liquid strengthen connections when co-activated
2. **Inter-module competition**: Different liquids compete through lateral inhibition
3. **Emergent specialization**: Skills naturally cluster into modules

### Modular Neural Network (MNN) Architecture

#### Self-Organizing MNN (Guo et al., 2023)
- Task decomposition layer
- Subnetwork layer (expert modules)
- Maximum modularity degree optimization
- Hub center identification in each sub-network

#### Key Design Principles
1. **Modularity Maximization**: Optimize for clear module boundaries
2. **Hub Centers**: Each module has a central organizing node
3. **Inter-module Connections**: Sparse connections between modules
4. **Reciprocal Connections**: Bidirectional information flow

### Integration with GN-CF Architecture

#### Mapping to Existing Components
- **Workspace (CortexFormer)**: Global broadcast layer (like cortical workspace)
- **Sheet (GyrusNet)**: 2D microcolumns with LE/LI connectivity
- **Liquids**: Self-organizing skill primitives using Hebbian clustering
- **Router**: Thalamic-like selection mechanism

#### Proposed Enhancements
1. **Ternary LE/LI Weights**: 
   - +1 for local excitation
   - -1 for lateral inhibition
   - 0 for no connection (sparsity)

2. **Hebbian Ternary Updates**:
   - If both neurons active: weight → +1 (strengthen)
   - If one active, one inhibited: weight → -1 (weaken)
   - Otherwise: weight → 0 (prune)

3. **Characteristic Wavelength**:
   - Define spatial scale for liquid clustering
   - Use ternary distance metrics


## Adversarial Curriculum and Dual-Role Self-Play

### Asymmetric Self-Play (Sukhbaatar et al., ICLR 2018) - 486 Citations

This seminal paper introduces a powerful paradigm that directly addresses the user's concept of "experiencer vs saboteur" sharing the same brain.

**Core Concept**: The approach splits the personality of a single agent into two parts (Alice and Bob) that share the same underlying model architecture:

1. **Alice (Teacher/Saboteur)**: Learns to generate goals/challenges that push Bob just past his current capabilities. Alice "proposes" tasks by doing a sequence of actions that Bob must undo or repeat.

2. **Bob (Student/Experiencer)**: Attempts to achieve the objectives set by Alice while also working on the original RL task.

**Key Insight**: Alice is incentivized to widen the gap between what she can do and what Bob can do, but pays a price for time taken - this balances the adversarial behavior and prevents Alice from creating impossible tasks.

**Why This Works**:
- Creates automatic curriculum of increasing difficulty
- Unsupervised exploration of the environment
- Reduces number of supervised episodes needed
- In some cases converges to higher reward than standard RL

### AMIGo: Adversarially Motivated Intrinsic Goals (Campero et al., ICLR 2021)

**Architecture**: A goal-generating "teacher" that proposes Adversarially Motivated Intrinsic Goals to train a goal-conditioned "student" policy.

**"Constructively Adversarial" Objective**: The teacher learns to propose increasingly challenging yet achievable goals. This creates a natural curriculum of self-proposed goals.

**Key Results**: Solves challenging procedurally-generated tasks where other forms of intrinsic motivation and state-of-the-art RL methods fail.

### User's Dual-Role Concept: Experiencer vs Saboteur (Same Brain)

The user's insight goes beyond standard asymmetric self-play by emphasizing that BOTH roles should use the SAME neural network, not just similar architectures. This creates:

1. **Shared Representation Learning**: The model must learn representations useful for both solving AND breaking problems.

2. **Self-Knowledge**: The saboteur knows exactly what the experiencer knows, preventing "cheap tricks."

3. **Genuine Understanding**: To sabotage effectively, the model must truly understand the problem space.

4. **Natural Difficulty Calibration**: The saboteur can only create challenges it could theoretically solve itself.

5. **Emergent Metacognition**: The model develops awareness of its own capabilities and limitations.

### Implementation for Balanced Ternary AGI

**Proposed Architecture**:

```
Single Neural Network (Balanced Ternary)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Experiencer  Saboteur
   Mode       Mode
    │         │
    └────┬────┘
         │
    Shared Weights
    Shared Liquids
    Shared Memory
```

**Mode Switching via Ternary Control Signal**:
- Mode = +1: Experiencer (solve the task)
- Mode = -1: Saboteur (create harder version)
- Mode = 0: Neutral/Evaluation (assess difficulty)

**Training Loop**:
1. Saboteur generates challenge (using same brain in -1 mode)
2. Experiencer attempts challenge (using same brain in +1 mode)
3. Compute regret: How much harder was it than expected?
4. Update shared weights to minimize regret while maintaining challenge generation

**Ternary Reward Signal**:
- +1: Experiencer succeeded, challenge was appropriate
- 0: Challenge was too easy (saboteur needs to try harder)
- -1: Challenge was impossible (saboteur was too aggressive)

### Connection to Affinity Liquids

The dual-role system naturally creates pressure for modular organization:

1. **Experiencer Liquids**: Specialize in solving specific task types
2. **Saboteur Liquids**: Specialize in finding weaknesses in those same areas
3. **Shared Foundation**: Both roles draw from the same liquid pool

This creates a natural adversarial curriculum where:
- Each liquid must be good at both solving AND challenging
- Liquids that can't do both get pruned
- New liquids emerge when existing ones can't handle novel challenges
