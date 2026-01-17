# GN-CF: GyrusNet-CortexFormer

**A Self-Organizing Neural Architecture with Balanced Ternary Computation and Adversarial Self-Play**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

GN-CF (GyrusNet-CortexFormer) is a novel neural architecture that explores an alternative path to AGI through **self-organizing principles** rather than massive parameter scaling. Inspired by neuroscience and cortical development, GN-CF combines several innovative concepts:

- **Balanced Ternary Neurons**: Weights in {-1, 0, +1} with quinary activations {-2, -1, 0, +1, +2} for efficient sparse computation
- **Self-Organizing Microcolumn Sheets**: 2D arrays with local excitation/lateral inhibition (LE/LI) dynamics
- **Affinity Liquids**: Skill primitives that enable emergent specialization into functional "lobes"
- **Mirror Generalization**: Unified forward/inverse models sharing a common representation
- **Adversarial Self-Play**: A single brain simultaneously plays solver (Experiencer) and challenger (Saboteur) roles

Unlike transformer-based approaches that scale through parameter count, GN-CF achieves emergent intelligence through **structural adaptation** - dynamically growing, pruning, and forming Hebbian-coupled modules based on task demands.

---

## Key Features

### 🧠 Neuroscience-Inspired Architecture
- Microcolumn organization inspired by cortical columns
- LE/LI connectivity patterns for self-organization
- Hebbian learning for team formation

### ⚡ Efficient Computation
- Ternary weights reduce memory and computation
- Sparse activations minimize unnecessary processing
- Dynamic resource allocation based on task complexity

### 🔄 Dual-Role Learning
- Experiencer mode: Learn to solve tasks
- Saboteur mode: Generate challenging scenarios
- Both roles share the same neural substrate

### 🎯 Emergent Modularity
- Skills cluster into functional regions without explicit programming
- Modules form through affinity-based self-organization
- Dynamic growth and pruning based on task demands

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SmartBrain                        │
│  ┌──────────────────────────────────────────────┐  │
│  │            CortexFormer (Workspace)          │  │
│  │  - Transformer-based state representation    │  │
│  │  - Generates router queries                  │  │
│  └──────────────────────────────────────────────┘  │
│                        ↓                            │
│  ┌──────────────────────────────────────────────┐  │
│  │              Router (Thalamic)               │  │
│  │  - Top-k microcolumn selection               │  │
│  │  - Hebbian coupling for team formation       │  │
│  └──────────────────────────────────────────────┘  │
│                        ↓                            │
│  ┌──────────────────────────────────────────────┐  │
│  │            GyrusNet (Sheet)                  │  │
│  │  - 2D microcolumn array                      │  │
│  │  - LE/LI connectivity                        │  │
│  │  - Affinity-based liquid mixing              │  │
│  └──────────────────────────────────────────────┘  │
│                        ↓                            │
│  ┌──────────────────────────────────────────────┐  │
│  │            Liquid Pool                       │  │
│  │  - Skill primitives                          │  │
│  │  - Dynamic spawning/merging                  │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/donttalktobrono/gn-cf-architecture.git
cd gn-cf-architecture

# Install dependencies
pip install numpy

# Optional: Install in development mode
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from smart_agi.smart_brain import SmartBrain
from smart_agi.envs.flappy_bird import FlappyBirdEnv

# Create environment
env = FlappyBirdEnv()

# Initialize brain
brain = SmartBrain(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    sheet_size=(8, 8),
    num_liquids=4
)

# Training loop
obs = env.reset()
for step in range(1000):
    action = brain.forward(obs)
    obs, reward, done, info = env.step(action)
    brain.backward(reward)
    
    if done:
        obs = env.reset()
```

### Adversarial Training

```python
from smart_agi.training.adversarial import AdversarialCurriculum

# Create adversarial curriculum
curriculum = AdversarialCurriculum(
    env=env,
    brain=brain,
    difficulty_range=(0.1, 1.0)
)

# Train with automatic difficulty adjustment
for episode in range(100):
    challenge = curriculum.generate_challenge()
    result = curriculum.run_episode(challenge)
    curriculum.update_difficulty(result)
```

---

## Project Structure

```
gn-cf-architecture/
├── smart_agi/
│   ├── core/              # Core neural components
│   │   ├── ternary.py     # Ternary arithmetic
│   │   ├── neuron.py      # Ternary neuron layers
│   │   ├── microcolumn.py # Microcolumn implementation
│   │   ├── sheet.py       # Self-organizing sheet
│   │   ├── router.py      # Hebbian routing
│   │   ├── growth.py      # Dynamic growth/pruning
│   │   └── mirror.py      # Mirror generalization
│   ├── liquids/           # Affinity liquid system
│   │   └── liquid.py      # Liquid pool and dynamics
│   ├── tools/             # Tool integration
│   │   └── tool_system.py # First-class tool actions
│   ├── training/          # Training systems
│   │   └── adversarial.py # Adversarial curriculum
│   ├── envs/              # Test environments
│   │   └── flappy_bird.py # Flappy Bird environment
│   ├── smart_brain.py     # Integrated brain
│   └── brain.py           # Legacy brain (reference)
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # Detailed architecture
│   └── RESEARCH.md        # Research notes
├── examples/              # Usage examples
│   └── train_flappy.py    # Training example
├── tests/                 # Unit tests
├── README.md              # This file
├── CITATION.cff           # Citation information
└── LICENSE                # MIT License
```

---

## Core Concepts

### Balanced Ternary Computation

Traditional neural networks use floating-point weights. GN-CF uses **balanced ternary** {-1, 0, +1} for weights and **quinary** {-2, -1, 0, +1, +2} for activations. This provides:

- **Efficiency**: Fewer bits per parameter
- **Sparsity**: Zero weights are truly zero (no computation)
- **Interpretability**: Discrete values are easier to understand

### Self-Organizing Sheets

The GyrusNet sheet uses **local excitation / lateral inhibition** (LE/LI) connectivity patterns inspired by cortical development. Nearby microcolumns excite each other, while distant ones inhibit, leading to natural clustering of related skills.

### Affinity Liquids

Instead of hardcoding skill types, GN-CF uses **affinity liquids** - abstract skill primitives that microcolumns can have varying affinities for. Over time, regions of the sheet develop high affinity for specific liquids, forming emergent functional "lobes."

### Mirror Generalization

The architecture includes a **forward/inverse model** that shares a common representation:
- **Forward mode**: Given state and action, predict next state
- **Inverse mode**: Given current and goal state, compute required action

This mirrors how biological brains handle both prediction and planning through the same neural substrate.

### Adversarial Self-Play

A single brain plays two roles:
- **Experiencer**: Tries to solve tasks and maximize reward
- **Saboteur**: Generates challenging scenarios to test the Experiencer

This prevents the system from exploiting simple tricks and forces genuine understanding.

---

## Experimental Results

### Self-Organization Metrics

The system demonstrates several key properties:

| Metric | Initial | After Training | Target |
|--------|---------|----------------|--------|
| Modularity | 0.25-0.35 | 0.30-0.40 | > 0.40 |
| Affinity Specialization | Random | Clustered | Clear regions |
| Mirror Consistency | Variable | Improving | > 0.90 |
| Dynamic Growth | Minimal | Task-dependent | Adaptive |

### Current Status

The architecture successfully demonstrates:
- ✅ Self-organizing sheet dynamics with LE/LI patterns
- ✅ Dynamic liquid formation and skill clustering
- ✅ Hebbian team formation in routing layer
- ✅ Adversarial curriculum with automatic difficulty adjustment
- ✅ Mirror model consistency between forward and inverse reasoning

**In Progress:**
- 🔄 Improving action selection for better task performance
- 🔄 Scaling to more complex environments
- 🔄 Hyperparameter tuning for optimal self-organization

---

## Documentation

- [**Architecture Design**](docs/ARCHITECTURE.md) - Detailed technical specification
- [**Research Notes**](docs/RESEARCH.md) - Theoretical foundations and references
- [**API Reference**](docs/API.md) - Complete API documentation

---

## Citation

If you use GN-CF in your research, please cite:

```bibtex
@software{guerrier2026gncf,
  author = {Guerrier, Jekhiel},
  title = {GN-CF: A Self-Organizing Neural Architecture with Balanced Ternary Computation and Adversarial Self-Play},
  year = {2026},
  url = {https://github.com/donttalktobrono/gn-cf-architecture}
}
```

Or use the `CITATION.cff` file for automatic citation generation.

---

## Contributing

This is currently a research project by an independent researcher. If you're interested in collaborating or have feedback:

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Pull Requests**: Contributions are welcome!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work was inspired by:
- **Numenta's Thousand Brains Theory** - Cortical column organization
- **Hierarchical Temporal Memory (HTM)** - Self-organizing sparse representations
- **Balanced Ternary Computing** - Efficient discrete computation
- **Adversarial Self-Play** - Robust learning through challenge generation

---

## Contact

**Jekhiel Guerrier**

For questions, collaboration opportunities, or feedback:
- GitHub: [@donttalktobrono](https://github.com/donttalktobrono)
- Project: [gn-cf-architecture](https://github.com/donttalktobrono/gn-cf-architecture)

---

## Roadmap

### Phase 1: Foundation (Current)
- [x] Core ternary neural network implementation
- [x] Self-organizing sheet with LE/LI
- [x] Affinity liquid system
- [x] Mirror generalization
- [x] Adversarial curriculum
- [ ] Improved action selection
- [ ] Comprehensive testing

### Phase 2: Scaling
- [ ] More complex environments (Atari, robotics)
- [ ] Multi-agent scenarios
- [ ] Tool use integration
- [ ] Memory systems

### Phase 3: Research
- [ ] Formal paper publication
- [ ] Benchmark comparisons
- [ ] Theoretical analysis
- [ ] Community building

---

**Built with curiosity, inspired by neuroscience, designed for the future of AI.**
