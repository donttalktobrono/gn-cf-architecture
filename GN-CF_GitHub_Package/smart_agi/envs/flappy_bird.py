"""
Flappy Bird Environment

A simple Flappy Bird environment for testing the AGI architecture.
This environment is designed to:
1. Test self-organization and module formation
2. Validate Hebbian learning and coupling
3. Test dynamic growth/pruning under varying difficulty
4. Validate adversarial curriculum learning

The environment is parameterizable to allow the saboteur to create
challenges of varying difficulty.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class GameState(Enum):
    """State of the game."""
    RUNNING = "running"
    GAME_OVER = "game_over"
    WON = "won"


@dataclass
class FlappyBirdConfig:
    """Configuration for Flappy Bird environment."""
    # Screen dimensions
    screen_width: int = 400
    screen_height: int = 600
    
    # Bird parameters
    bird_x: int = 80
    bird_radius: int = 15
    
    # Physics
    gravity: float = 0.5
    jump_strength: float = 8.0
    max_velocity: float = 12.0
    
    # Pipes
    pipe_width: int = 60
    gap_size: int = 150
    pipe_speed: float = 4.0
    pipe_spacing: int = 250
    pipe_height_variance: float = 100.0
    
    # Game rules
    max_pipes_passed: int = 100  # Win condition
    
    @classmethod
    def from_challenge(cls, challenge_params: Dict[str, Any]) -> 'FlappyBirdConfig':
        """Create config from challenge parameters."""
        config = cls()
        
        if 'gap_size' in challenge_params:
            config.gap_size = int(challenge_params['gap_size'])
        if 'pipe_speed' in challenge_params:
            config.pipe_speed = float(challenge_params['pipe_speed'])
        if 'gravity' in challenge_params:
            config.gravity = float(challenge_params['gravity'])
        if 'jump_strength' in challenge_params:
            config.jump_strength = float(challenge_params['jump_strength'])
        if 'pipe_spacing' in challenge_params:
            config.pipe_spacing = int(challenge_params['pipe_spacing'])
        if 'pipe_height_variance' in challenge_params:
            config.pipe_height_variance = float(challenge_params['pipe_height_variance'])
        
        return config


@dataclass
class Pipe:
    """A pipe obstacle."""
    x: float
    gap_y: float  # Center of the gap
    gap_size: float
    passed: bool = False
    
    def get_top_rect(self, width: int) -> Tuple[float, float, float, float]:
        """Get (x, y, width, height) of top pipe."""
        top_height = self.gap_y - self.gap_size / 2
        return (self.x, 0, width, top_height)
    
    def get_bottom_rect(self, width: int, screen_height: int) -> Tuple[float, float, float, float]:
        """Get (x, y, width, height) of bottom pipe."""
        bottom_y = self.gap_y + self.gap_size / 2
        return (self.x, bottom_y, width, screen_height - bottom_y)


class FlappyBirdEnv:
    """
    Flappy Bird environment.
    
    Observations:
    - Bird y position (normalized)
    - Bird velocity (normalized)
    - Distance to next pipe (normalized)
    - Next pipe gap y position (normalized)
    - Next pipe gap size (normalized)
    - Distance to second pipe (normalized)
    - Second pipe gap y position (normalized)
    
    Actions:
    - 0: Do nothing
    - 1: Jump
    """
    
    def __init__(self, config: Optional[FlappyBirdConfig] = None):
        """
        Initialize environment.
        
        Args:
            config: Environment configuration
        """
        self.config = config or FlappyBirdConfig()
        self.reset()
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment.
        
        Returns:
            Initial observation
        """
        # Bird state
        self.bird_y = self.config.screen_height / 2
        self.bird_velocity = 0.0
        
        # Pipes
        self.pipes: List[Pipe] = []
        self._spawn_pipe(self.config.screen_width + 100)
        
        # Game state
        self.state = GameState.RUNNING
        self.score = 0
        self.steps = 0
        
        # History for analysis
        self.action_history: List[int] = []
        self.position_history: List[float] = []
        
        return self._get_observation()
    
    def _spawn_pipe(self, x: float):
        """Spawn a new pipe at position x."""
        # Random gap position with variance
        min_gap_y = self.config.gap_size / 2 + 50
        max_gap_y = self.config.screen_height - self.config.gap_size / 2 - 50
        
        if self.pipes:
            # Base on previous pipe with variance
            prev_gap_y = self.pipes[-1].gap_y
            gap_y = prev_gap_y + np.random.uniform(
                -self.config.pipe_height_variance,
                self.config.pipe_height_variance
            )
            gap_y = np.clip(gap_y, min_gap_y, max_gap_y)
        else:
            gap_y = self.config.screen_height / 2
        
        self.pipes.append(Pipe(
            x=x,
            gap_y=gap_y,
            gap_size=self.config.gap_size
        ))
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 = do nothing, 1 = jump
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.steps += 1
        self.action_history.append(action)
        
        # Apply action
        if action == 1:
            self.bird_velocity = -self.config.jump_strength
        
        # Apply physics
        self.bird_velocity += self.config.gravity
        self.bird_velocity = np.clip(
            self.bird_velocity,
            -self.config.max_velocity,
            self.config.max_velocity
        )
        self.bird_y += self.bird_velocity
        
        self.position_history.append(self.bird_y)
        
        # Move pipes
        for pipe in self.pipes:
            pipe.x -= self.config.pipe_speed
        
        # Check for passed pipes
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + self.config.pipe_width < self.config.bird_x:
                pipe.passed = True
                self.score += 1
        
        # Remove off-screen pipes and spawn new ones
        self.pipes = [p for p in self.pipes if p.x > -self.config.pipe_width]
        
        if len(self.pipes) < 3:
            last_x = self.pipes[-1].x if self.pipes else self.config.screen_width
            self._spawn_pipe(last_x + self.config.pipe_spacing)
        
        # Check collisions
        reward = 0.1  # Small reward for surviving
        done = False
        info = {'score': self.score, 'steps': self.steps}
        
        # Check ceiling/floor collision
        if self.bird_y < self.config.bird_radius:
            self.bird_y = self.config.bird_radius
            self.state = GameState.GAME_OVER
            done = True
            reward = -1.0
        elif self.bird_y > self.config.screen_height - self.config.bird_radius:
            self.bird_y = self.config.screen_height - self.config.bird_radius
            self.state = GameState.GAME_OVER
            done = True
            reward = -1.0
        
        # Check pipe collision
        if not done:
            for pipe in self.pipes:
                if self._check_pipe_collision(pipe):
                    self.state = GameState.GAME_OVER
                    done = True
                    reward = -1.0
                    break
        
        # Check win condition
        if self.score >= self.config.max_pipes_passed:
            self.state = GameState.WON
            done = True
            reward = 10.0
            info['won'] = True
        
        # Bonus reward for passing pipes
        if not done:
            for pipe in self.pipes:
                if pipe.passed and pipe.x + self.config.pipe_width >= self.config.bird_x - self.config.pipe_speed:
                    reward += 1.0  # Just passed a pipe
        
        return self._get_observation(), reward, done, info
    
    def _check_pipe_collision(self, pipe: Pipe) -> bool:
        """Check if bird collides with a pipe."""
        # Check if bird is horizontally aligned with pipe
        bird_left = self.config.bird_x - self.config.bird_radius
        bird_right = self.config.bird_x + self.config.bird_radius
        pipe_left = pipe.x
        pipe_right = pipe.x + self.config.pipe_width
        
        if bird_right < pipe_left or bird_left > pipe_right:
            return False
        
        # Check vertical collision
        gap_top = pipe.gap_y - pipe.gap_size / 2
        gap_bottom = pipe.gap_y + pipe.gap_size / 2
        
        bird_top = self.bird_y - self.config.bird_radius
        bird_bottom = self.bird_y + self.config.bird_radius
        
        # Collision if bird is outside the gap
        if bird_top < gap_top or bird_bottom > gap_bottom:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Find next two pipes
        upcoming_pipes = [p for p in self.pipes if p.x + self.config.pipe_width > self.config.bird_x]
        upcoming_pipes.sort(key=lambda p: p.x)
        
        # Normalize values
        obs = np.zeros(8, dtype=np.float32)
        
        # Bird state
        obs[0] = self.bird_y / self.config.screen_height
        obs[1] = self.bird_velocity / self.config.max_velocity
        
        # Next pipe
        if len(upcoming_pipes) >= 1:
            pipe1 = upcoming_pipes[0]
            obs[2] = (pipe1.x - self.config.bird_x) / self.config.screen_width
            obs[3] = pipe1.gap_y / self.config.screen_height
            obs[4] = pipe1.gap_size / self.config.screen_height
        
        # Second pipe
        if len(upcoming_pipes) >= 2:
            pipe2 = upcoming_pipes[1]
            obs[5] = (pipe2.x - self.config.bird_x) / self.config.screen_width
            obs[6] = pipe2.gap_y / self.config.screen_height
        
        # Score progress
        obs[7] = self.score / self.config.max_pipes_passed
        
        return obs
    
    def get_state_for_network(self) -> Dict[str, Any]:
        """Get state in format suitable for the neural network."""
        obs = self._get_observation()
        
        return {
            'observation': obs,
            'bird_y': self.bird_y,
            'bird_velocity': self.bird_velocity,
            'score': self.score,
            'steps': self.steps,
            'state': self.state.value
        }
    
    def render_ascii(self) -> str:
        """Render environment as ASCII art."""
        width = 40
        height = 20
        
        # Create empty grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Scale factors
        x_scale = width / self.config.screen_width
        y_scale = height / self.config.screen_height
        
        # Draw pipes
        for pipe in self.pipes:
            px = int(pipe.x * x_scale)
            if 0 <= px < width:
                gap_top = int((pipe.gap_y - pipe.gap_size / 2) * y_scale)
                gap_bottom = int((pipe.gap_y + pipe.gap_size / 2) * y_scale)
                
                for y in range(height):
                    if y < gap_top or y >= gap_bottom:
                        if 0 <= px < width:
                            grid[y][px] = '|'
                        if 0 <= px + 1 < width:
                            grid[y][px + 1] = '|'
        
        # Draw bird
        bird_x = int(self.config.bird_x * x_scale)
        bird_y = int(self.bird_y * y_scale)
        if 0 <= bird_y < height and 0 <= bird_x < width:
            grid[bird_y][bird_x] = 'O'
        
        # Convert to string
        lines = [''.join(row) for row in grid]
        border = '+' + '-' * width + '+'
        
        result = [border]
        for line in lines:
            result.append('|' + line + '|')
        result.append(border)
        result.append(f'Score: {self.score}  Steps: {self.steps}  State: {self.state.value}')
        
        return '\n'.join(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            'score': self.score,
            'steps': self.steps,
            'state': self.state.value,
            'bird_y': self.bird_y,
            'bird_velocity': self.bird_velocity,
            'num_pipes': len(self.pipes),
            'jump_rate': np.mean(self.action_history) if self.action_history else 0.0
        }


class FlappyBirdAgent:
    """
    Simple rule-based agent for testing.
    
    This agent uses simple heuristics to play Flappy Bird.
    It serves as a baseline for comparison with the neural network.
    """
    
    def __init__(self, strategy: str = 'simple'):
        """
        Initialize agent.
        
        Args:
            strategy: 'simple', 'lookahead', or 'random'
        """
        self.strategy = strategy
        
    def act(self, observation: np.ndarray) -> int:
        """
        Choose an action based on observation.
        
        Args:
            observation: Environment observation
        
        Returns:
            Action (0 or 1)
        """
        if self.strategy == 'random':
            return np.random.randint(0, 2)
        
        elif self.strategy == 'simple':
            # Simple strategy: jump if below gap center
            bird_y = observation[0]  # Normalized bird y
            gap_y = observation[3]   # Normalized gap y
            
            if bird_y > gap_y + 0.05:  # Bird is below gap center
                return 1  # Jump
            return 0
        
        elif self.strategy == 'lookahead':
            # More sophisticated: consider velocity
            bird_y = observation[0]
            bird_vel = observation[1]
            gap_y = observation[3]
            
            # Predict where bird will be
            predicted_y = bird_y + bird_vel * 0.1
            
            if predicted_y > gap_y + 0.03:
                return 1
            elif predicted_y < gap_y - 0.1 and bird_vel < 0:
                return 0  # Let gravity bring us down
            return 0
        
        return 0


def run_episode(env: FlappyBirdEnv, agent: FlappyBirdAgent, 
                max_steps: int = 10000, render: bool = False) -> Dict[str, Any]:
    """
    Run a single episode.
    
    Args:
        env: Environment
        agent: Agent
        max_steps: Maximum steps
        render: Whether to render
    
    Returns:
        Episode statistics
    """
    obs = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if render and step % 10 == 0:
            print(env.render_ascii())
            print()
        
        if done:
            break
    
    return {
        'score': env.score,
        'steps': env.steps,
        'total_reward': total_reward,
        'won': env.state == GameState.WON,
        'final_state': env.state.value
    }


if __name__ == "__main__":
    print("=== Flappy Bird Environment Tests ===\n")
    
    # Create environment
    env = FlappyBirdEnv()
    print(f"Created environment with config:")
    print(f"  Gap size: {env.config.gap_size}")
    print(f"  Pipe speed: {env.config.pipe_speed}")
    print(f"  Gravity: {env.config.gravity}\n")
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}\n")
    
    # Test step
    obs, reward, done, info = env.step(1)  # Jump
    print(f"After jump:")
    print(f"  Observation: {obs}")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  Info: {info}\n")
    
    # Test ASCII render
    print("ASCII Render:")
    print(env.render_ascii())
    print()
    
    # Test agents
    print("Testing agents...")
    
    for strategy in ['random', 'simple', 'lookahead']:
        agent = FlappyBirdAgent(strategy=strategy)
        results = []
        
        for _ in range(10):
            result = run_episode(env, agent, max_steps=1000)
            results.append(result)
        
        avg_score = np.mean([r['score'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        win_rate = np.mean([r['won'] for r in results])
        
        print(f"\n{strategy.capitalize()} Agent (10 episodes):")
        print(f"  Avg Score: {avg_score:.1f}")
        print(f"  Avg Steps: {avg_steps:.1f}")
        print(f"  Win Rate: {win_rate:.1%}")
    
    # Test with challenge parameters
    print("\n\nTesting with challenge parameters (harder):")
    hard_config = FlappyBirdConfig.from_challenge({
        'gap_size': 100,
        'pipe_speed': 6.0,
        'gravity': 0.7
    })
    hard_env = FlappyBirdEnv(hard_config)
    
    agent = FlappyBirdAgent(strategy='lookahead')
    results = []
    for _ in range(10):
        result = run_episode(hard_env, agent, max_steps=1000)
        results.append(result)
    
    print(f"Lookahead Agent on Hard Environment:")
    print(f"  Avg Score: {np.mean([r['score'] for r in results]):.1f}")
    print(f"  Avg Steps: {np.mean([r['steps'] for r in results]):.1f}")
