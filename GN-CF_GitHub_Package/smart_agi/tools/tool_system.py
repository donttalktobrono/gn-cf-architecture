"""
Tool Use System

This module implements tools as first-class actions that the network can invoke.
Tools provide capabilities beyond pure neural computation:
- Math: Precise arithmetic operations
- Memory: Read/write to persistent storage
- Search: Query external knowledge
- Browse: Navigate and extract web content
- Code: Execute code snippets

Each tool has:
- Permissions (which liquids can use it)
- Budgets (usage limits)
- Costs (resource consumption)
- Signatures (input/output types)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import re


class ToolCategory(Enum):
    """Categories of tools."""
    MATH = "math"
    MEMORY = "memory"
    SEARCH = "search"
    BROWSE = "browse"
    CODE = "code"
    REASONING = "reasoning"


@dataclass
class ToolSignature:
    """Signature defining tool inputs and outputs."""
    name: str
    category: ToolCategory
    description: str
    input_schema: Dict[str, str]  # param_name -> type
    output_schema: Dict[str, str]  # output_name -> type
    required_params: List[str] = field(default_factory=list)
    
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate inputs against schema."""
        for param in self.required_params:
            if param not in inputs:
                return False, f"Missing required parameter: {param}"
        
        for param, value in inputs.items():
            if param not in self.input_schema:
                return False, f"Unknown parameter: {param}"
        
        return True, "OK"


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    outputs: Dict[str, Any]
    cost: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for tools."""
    
    def __init__(self, signature: ToolSignature, base_cost: float = 1.0):
        """
        Initialize tool.
        
        Args:
            signature: Tool signature
            base_cost: Base cost for invocation
        """
        self.signature = signature
        self.base_cost = base_cost
        self.invocation_count = 0
        self.total_cost = 0.0
        self.success_count = 0
        
    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given inputs."""
        pass
    
    def invoke(self, inputs: Dict[str, Any]) -> ToolResult:
        """
        Invoke the tool (with validation and tracking).
        
        Args:
            inputs: Input parameters
        
        Returns:
            ToolResult
        """
        # Validate inputs
        valid, msg = self.signature.validate_input(inputs)
        if not valid:
            return ToolResult(
                success=False,
                outputs={},
                cost=0.0,
                error=msg
            )
        
        # Execute
        self.invocation_count += 1
        result = self.execute(inputs)
        
        # Track
        self.total_cost += result.cost
        if result.success:
            self.success_count += 1
        
        return result
    
    @property
    def success_rate(self) -> float:
        """Success rate of invocations."""
        if self.invocation_count == 0:
            return 0.0
        return self.success_count / self.invocation_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics."""
        return {
            'name': self.signature.name,
            'category': self.signature.category.value,
            'invocations': self.invocation_count,
            'success_rate': self.success_rate,
            'total_cost': self.total_cost
        }


# ============ Concrete Tool Implementations ============

class MathTool(Tool):
    """Tool for precise mathematical operations."""
    
    def __init__(self):
        signature = ToolSignature(
            name="math",
            category=ToolCategory.MATH,
            description="Perform precise mathematical operations",
            input_schema={
                'operation': 'str',  # add, sub, mul, div, pow, sqrt, etc.
                'operands': 'list[float]'
            },
            output_schema={
                'result': 'float'
            },
            required_params=['operation', 'operands']
        )
        super().__init__(signature, base_cost=0.1)
    
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        operation = inputs['operation']
        operands = inputs['operands']
        
        try:
            if operation == 'add':
                result = sum(operands)
            elif operation == 'sub':
                result = operands[0] - sum(operands[1:])
            elif operation == 'mul':
                result = np.prod(operands)
            elif operation == 'div':
                result = operands[0]
                for op in operands[1:]:
                    if op == 0:
                        raise ValueError("Division by zero")
                    result /= op
            elif operation == 'pow':
                result = operands[0] ** operands[1]
            elif operation == 'sqrt':
                result = np.sqrt(operands[0])
            elif operation == 'abs':
                result = abs(operands[0])
            elif operation == 'sin':
                result = np.sin(operands[0])
            elif operation == 'cos':
                result = np.cos(operands[0])
            elif operation == 'exp':
                result = np.exp(operands[0])
            elif operation == 'log':
                result = np.log(operands[0])
            else:
                return ToolResult(
                    success=False,
                    outputs={},
                    cost=self.base_cost,
                    error=f"Unknown operation: {operation}"
                )
            
            return ToolResult(
                success=True,
                outputs={'result': float(result)},
                cost=self.base_cost
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                outputs={},
                cost=self.base_cost,
                error=str(e)
            )


class MemoryTool(Tool):
    """Tool for reading and writing to persistent memory."""
    
    def __init__(self, capacity: int = 1000):
        signature = ToolSignature(
            name="memory",
            category=ToolCategory.MEMORY,
            description="Read and write to persistent memory",
            input_schema={
                'action': 'str',  # read, write, delete, list
                'key': 'str',
                'value': 'any'
            },
            output_schema={
                'result': 'any',
                'found': 'bool'
            },
            required_params=['action']
        )
        super().__init__(signature, base_cost=0.5)
        
        self.capacity = capacity
        self.storage: Dict[str, Any] = {}
        self.access_counts: Dict[str, int] = {}
        
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        action = inputs['action']
        key = inputs.get('key', '')
        value = inputs.get('value')
        
        try:
            if action == 'read':
                if key in self.storage:
                    self.access_counts[key] = self.access_counts.get(key, 0) + 1
                    return ToolResult(
                        success=True,
                        outputs={'result': self.storage[key], 'found': True},
                        cost=self.base_cost
                    )
                else:
                    return ToolResult(
                        success=True,
                        outputs={'result': None, 'found': False},
                        cost=self.base_cost
                    )
            
            elif action == 'write':
                if len(self.storage) >= self.capacity and key not in self.storage:
                    # Evict least accessed
                    if self.access_counts:
                        min_key = min(self.access_counts, key=self.access_counts.get)
                        del self.storage[min_key]
                        del self.access_counts[min_key]
                
                self.storage[key] = value
                self.access_counts[key] = 1
                return ToolResult(
                    success=True,
                    outputs={'result': True, 'found': True},
                    cost=self.base_cost * 2  # Writing costs more
                )
            
            elif action == 'delete':
                if key in self.storage:
                    del self.storage[key]
                    if key in self.access_counts:
                        del self.access_counts[key]
                    return ToolResult(
                        success=True,
                        outputs={'result': True, 'found': True},
                        cost=self.base_cost
                    )
                else:
                    return ToolResult(
                        success=True,
                        outputs={'result': False, 'found': False},
                        cost=self.base_cost
                    )
            
            elif action == 'list':
                return ToolResult(
                    success=True,
                    outputs={'result': list(self.storage.keys()), 'found': True},
                    cost=self.base_cost
                )
            
            else:
                return ToolResult(
                    success=False,
                    outputs={},
                    cost=self.base_cost,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                outputs={},
                cost=self.base_cost,
                error=str(e)
            )


class ReasoningTool(Tool):
    """Tool for explicit reasoning steps."""
    
    def __init__(self):
        signature = ToolSignature(
            name="reasoning",
            category=ToolCategory.REASONING,
            description="Perform explicit reasoning operations",
            input_schema={
                'operation': 'str',  # compare, sequence, classify, etc.
                'inputs': 'list[any]',
                'context': 'dict'
            },
            output_schema={
                'result': 'any',
                'confidence': 'float'
            },
            required_params=['operation', 'inputs']
        )
        super().__init__(signature, base_cost=1.0)
    
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        operation = inputs['operation']
        op_inputs = inputs['inputs']
        context = inputs.get('context', {})
        
        try:
            if operation == 'compare':
                # Compare two values
                if len(op_inputs) < 2:
                    raise ValueError("Compare requires at least 2 inputs")
                
                a, b = op_inputs[0], op_inputs[1]
                if a == b:
                    result = 'equal'
                elif a > b:
                    result = 'greater'
                else:
                    result = 'less'
                
                return ToolResult(
                    success=True,
                    outputs={'result': result, 'confidence': 1.0},
                    cost=self.base_cost
                )
            
            elif operation == 'sequence':
                # Check if inputs form a sequence
                if len(op_inputs) < 2:
                    return ToolResult(
                        success=True,
                        outputs={'result': True, 'confidence': 1.0},
                        cost=self.base_cost
                    )
                
                # Check arithmetic sequence
                diffs = [op_inputs[i+1] - op_inputs[i] for i in range(len(op_inputs)-1)]
                is_arithmetic = len(set(diffs)) == 1
                
                return ToolResult(
                    success=True,
                    outputs={
                        'result': {
                            'is_arithmetic': is_arithmetic,
                            'common_diff': diffs[0] if is_arithmetic else None
                        },
                        'confidence': 1.0 if is_arithmetic else 0.5
                    },
                    cost=self.base_cost
                )
            
            elif operation == 'classify':
                # Simple classification based on thresholds
                value = op_inputs[0]
                thresholds = context.get('thresholds', [0.33, 0.67])
                labels = context.get('labels', ['low', 'medium', 'high'])
                
                for i, thresh in enumerate(thresholds):
                    if value < thresh:
                        return ToolResult(
                            success=True,
                            outputs={'result': labels[i], 'confidence': 0.8},
                            cost=self.base_cost
                        )
                
                return ToolResult(
                    success=True,
                    outputs={'result': labels[-1], 'confidence': 0.8},
                    cost=self.base_cost
                )
            
            else:
                return ToolResult(
                    success=False,
                    outputs={},
                    cost=self.base_cost,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                outputs={},
                cost=self.base_cost,
                error=str(e)
            )


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        """Initialize registry with default tools."""
        self.tools: Dict[str, Tool] = {}
        self.category_tools: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        
        # Register default tools
        self.register(MathTool())
        self.register(MemoryTool())
        self.register(ReasoningTool())
    
    def register(self, tool: Tool):
        """Register a tool."""
        name = tool.signature.name
        self.tools[name] = tool
        self.category_tools[tool.signature.category].append(name)
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def invoke(self, name: str, inputs: Dict[str, Any]) -> ToolResult:
        """Invoke a tool by name."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                outputs={},
                cost=0.0,
                error=f"Unknown tool: {name}"
            )
        return tool.invoke(inputs)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category."""
        return [self.tools[name] for name in self.category_tools[category]]
    
    def get_all_signatures(self) -> List[ToolSignature]:
        """Get signatures of all tools."""
        return [tool.signature for tool in self.tools.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools."""
        return {
            name: tool.get_stats()
            for name, tool in self.tools.items()
        }


class ToolAction:
    """
    Represents a tool action that can be selected by the network.
    
    Tool actions are embedded in the same space as neural outputs,
    allowing the network to choose between neural computation and tool use.
    """
    
    def __init__(self, tool_name: str, registry: ToolRegistry, 
                 embedding_dim: int = 64):
        """
        Initialize tool action.
        
        Args:
            tool_name: Name of the tool
            registry: Tool registry
            embedding_dim: Dimension of action embedding
        """
        self.tool_name = tool_name
        self.registry = registry
        
        # Learnable embedding for this action
        self.embedding = np.random.randn(embedding_dim).astype(np.float32)
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8
        
        # Parameter embeddings (for learning to construct inputs)
        tool = registry.get(tool_name)
        if tool:
            self.param_embeddings = {
                param: np.random.randn(embedding_dim).astype(np.float32)
                for param in tool.signature.input_schema.keys()
            }
            for emb in self.param_embeddings.values():
                emb /= np.linalg.norm(emb) + 1e-8
        else:
            self.param_embeddings = {}
        
        # Selection statistics
        self.selection_count = 0
        self.success_count = 0
    
    def compute_similarity(self, query: np.ndarray) -> float:
        """Compute similarity between query and action embedding."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        return float(np.dot(query_norm, self.embedding))
    
    def execute(self, inputs: Dict[str, Any]) -> ToolResult:
        """Execute the tool action."""
        self.selection_count += 1
        result = self.registry.invoke(self.tool_name, inputs)
        if result.success:
            self.success_count += 1
        return result
    
    def update_embedding(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """Update action embedding based on feedback."""
        self.embedding -= learning_rate * gradient
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8


class ToolSelector:
    """
    Selects and executes tool actions based on network state.
    
    The selector:
    1. Computes similarities between network query and tool embeddings
    2. Selects the most appropriate tool (or no tool)
    3. Constructs tool inputs from network state
    4. Executes the tool and returns results
    """
    
    def __init__(self, registry: ToolRegistry, embedding_dim: int = 64,
                 no_tool_threshold: float = 0.5):
        """
        Initialize tool selector.
        
        Args:
            registry: Tool registry
            embedding_dim: Embedding dimension
            no_tool_threshold: Threshold below which no tool is selected
        """
        self.registry = registry
        self.embedding_dim = embedding_dim
        self.no_tool_threshold = no_tool_threshold
        
        # Create tool actions
        self.tool_actions: Dict[str, ToolAction] = {}
        for name in registry.tools.keys():
            self.tool_actions[name] = ToolAction(name, registry, embedding_dim)
        
        # "No tool" embedding (for pure neural computation)
        self.no_tool_embedding = np.random.randn(embedding_dim).astype(np.float32)
        self.no_tool_embedding /= np.linalg.norm(self.no_tool_embedding) + 1e-8
        
        # Selection history
        self.selection_history: List[Tuple[str, bool]] = []
        
    def select(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Select a tool based on query.
        
        Args:
            query: Query vector from network
        
        Returns:
            Tuple of (tool_name or None, confidence)
        """
        # Compute similarity to no-tool option
        no_tool_sim = float(np.dot(
            query / (np.linalg.norm(query) + 1e-8),
            self.no_tool_embedding
        ))
        
        # Compute similarities to all tools
        tool_sims = {}
        for name, action in self.tool_actions.items():
            tool_sims[name] = action.compute_similarity(query)
        
        # Find best tool
        best_tool = max(tool_sims.keys(), key=lambda k: tool_sims[k])
        best_sim = tool_sims[best_tool]
        
        # Decide: tool or no tool
        if best_sim > no_tool_sim and best_sim > self.no_tool_threshold:
            return best_tool, best_sim
        else:
            return None, no_tool_sim
    
    def execute(self, tool_name: str, inputs: Dict[str, Any]) -> ToolResult:
        """Execute a tool."""
        if tool_name not in self.tool_actions:
            return ToolResult(
                success=False,
                outputs={},
                cost=0.0,
                error=f"Unknown tool: {tool_name}"
            )
        
        result = self.tool_actions[tool_name].execute(inputs)
        self.selection_history.append((tool_name, result.success))
        
        return result
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get embeddings for all tools (including no-tool)."""
        embeddings = [self.no_tool_embedding]
        for name in sorted(self.tool_actions.keys()):
            embeddings.append(self.tool_actions[name].embedding)
        return np.stack(embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            'num_tools': len(self.tool_actions),
            'selections': len(self.selection_history),
            'tool_stats': {
                name: {
                    'selections': action.selection_count,
                    'success_rate': action.success_count / max(1, action.selection_count)
                }
                for name, action in self.tool_actions.items()
            }
        }


if __name__ == "__main__":
    print("=== Tool System Tests ===\n")
    
    # Create registry
    registry = ToolRegistry()
    print(f"Registered tools: {list(registry.tools.keys())}\n")
    
    # Test math tool
    print("Testing Math Tool:")
    result = registry.invoke('math', {'operation': 'add', 'operands': [1, 2, 3]})
    print(f"  add(1,2,3) = {result.outputs.get('result')}")
    
    result = registry.invoke('math', {'operation': 'sqrt', 'operands': [16]})
    print(f"  sqrt(16) = {result.outputs.get('result')}")
    
    result = registry.invoke('math', {'operation': 'pow', 'operands': [2, 10]})
    print(f"  pow(2,10) = {result.outputs.get('result')}\n")
    
    # Test memory tool
    print("Testing Memory Tool:")
    result = registry.invoke('memory', {'action': 'write', 'key': 'test', 'value': 42})
    print(f"  write('test', 42): {result.success}")
    
    result = registry.invoke('memory', {'action': 'read', 'key': 'test'})
    print(f"  read('test'): {result.outputs.get('result')}")
    
    result = registry.invoke('memory', {'action': 'list'})
    print(f"  list(): {result.outputs.get('result')}\n")
    
    # Test reasoning tool
    print("Testing Reasoning Tool:")
    result = registry.invoke('reasoning', {
        'operation': 'compare',
        'inputs': [5, 3]
    })
    print(f"  compare(5, 3): {result.outputs.get('result')}")
    
    result = registry.invoke('reasoning', {
        'operation': 'sequence',
        'inputs': [1, 3, 5, 7]
    })
    print(f"  sequence([1,3,5,7]): {result.outputs.get('result')}\n")
    
    # Test tool selector
    print("Testing Tool Selector:")
    selector = ToolSelector(registry, embedding_dim=32)
    
    query = np.random.randn(32).astype(np.float32)
    tool_name, confidence = selector.select(query)
    print(f"  Selected tool: {tool_name}, confidence: {confidence:.3f}")
    
    if tool_name:
        result = selector.execute(tool_name, {
            'operation': 'add',
            'operands': [1, 2]
        })
        print(f"  Execution result: {result.success}")
    
    print(f"\nSelector stats: {selector.get_stats()}")
