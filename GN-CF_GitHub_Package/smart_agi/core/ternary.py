"""
Balanced Ternary Arithmetic Module

This module implements balanced ternary arithmetic operations for neural network
computations. Balanced ternary uses digits {-1, 0, +1} (often written as {T, 0, 1})
and provides natural representation of negative numbers without a separate sign.

Key advantages for neural networks:
- Multiplication by weights is trivial (sign flip, zero, or identity)
- Natural sparsity (0 = no connection)
- Biologically plausible (inhibition/excitation)
- Hardware efficient (simple logic operations)
"""

import numpy as np
from typing import Union, List, Tuple
from enum import IntEnum


class Trit(IntEnum):
    """Balanced ternary digit (trit): -1, 0, or +1"""
    NEG = -1  # Also written as 'T'
    ZERO = 0
    POS = 1


class Quint(IntEnum):
    """Balanced quinary digit: -2, -1, 0, +1, or +2"""
    STRONG_NEG = -2
    WEAK_NEG = -1
    ZERO = 0
    WEAK_POS = 1
    STRONG_POS = 2


def quantize_to_ternary(value: float, threshold: float = 0.33) -> int:
    """
    Quantize a continuous value to balanced ternary {-1, 0, +1}.
    
    Args:
        value: Input value (typically in range [-1, 1])
        threshold: Threshold for non-zero assignment
    
    Returns:
        Ternary value: -1, 0, or +1
    """
    if value > threshold:
        return 1
    elif value < -threshold:
        return -1
    else:
        return 0


def quantize_to_quinary(value: float) -> int:
    """
    Quantize a continuous value to balanced quinary {-2, -1, 0, +1, +2}.
    
    Args:
        value: Input value (typically in range [-2, 2] or normalized)
    
    Returns:
        Quinary value: -2, -1, 0, +1, or +2
    """
    if value <= -1.5:
        return -2
    elif value <= -0.5:
        return -1
    elif value <= 0.5:
        return 0
    elif value <= 1.5:
        return 1
    else:
        return 2


def ternary_multiply(a: int, b: int) -> int:
    """
    Multiply two ternary values.
    
    This is extremely efficient:
    - If either is 0, result is 0
    - If signs match, result is +1
    - If signs differ, result is -1
    
    Args:
        a: First ternary value {-1, 0, +1}
        b: Second ternary value {-1, 0, +1}
    
    Returns:
        Product (also ternary): {-1, 0, +1}
    """
    if a == 0 or b == 0:
        return 0
    return 1 if a == b else -1


def ternary_dot_product(weights: np.ndarray, inputs: np.ndarray) -> float:
    """
    Compute dot product with ternary weights.
    
    This is highly efficient because multiplication is trivial:
    - w=0: skip (no contribution)
    - w=+1: add input
    - w=-1: subtract input
    
    Args:
        weights: Array of ternary weights {-1, 0, +1}
        inputs: Array of input values (any numeric type)
    
    Returns:
        Weighted sum
    """
    result = 0.0
    for w, x in zip(weights, inputs):
        if w == 1:
            result += x
        elif w == -1:
            result -= x
        # w == 0: no contribution
    return result


def ternary_dot_product_vectorized(weights: np.ndarray, inputs: np.ndarray) -> float:
    """
    Vectorized ternary dot product using NumPy.
    
    Args:
        weights: Array of ternary weights {-1, 0, +1}
        inputs: Array of input values
    
    Returns:
        Weighted sum
    """
    # This is equivalent to np.dot but emphasizes the ternary nature
    # In practice, NumPy handles this efficiently
    return np.sum(weights * inputs)


def ternary_matrix_multiply(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Matrix-vector multiplication with ternary weight matrix.
    
    Args:
        W: Ternary weight matrix (m x n) with values in {-1, 0, +1}
        x: Input vector (n,)
    
    Returns:
        Output vector (m,)
    """
    return W @ x


class TernaryArray:
    """
    A specialized array class for ternary values with efficient operations.
    
    Stores values as int8 internally for memory efficiency while ensuring
    all values remain in {-1, 0, +1}.
    """
    
    def __init__(self, data: Union[np.ndarray, List, Tuple], copy: bool = True):
        """
        Initialize ternary array.
        
        Args:
            data: Input data (will be quantized to ternary)
            copy: Whether to copy the data
        """
        if isinstance(data, np.ndarray):
            arr = data.copy() if copy else data
        else:
            arr = np.array(data)
        
        # Quantize to ternary
        self._data = np.clip(np.round(arr), -1, 1).astype(np.int8)
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def shape(self) -> Tuple:
        return self._data.shape
    
    @property
    def sparsity(self) -> float:
        """Fraction of zero elements"""
        return np.mean(self._data == 0)
    
    def dot(self, other: np.ndarray) -> np.ndarray:
        """Dot product with another array"""
        return self._data @ other
    
    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.dot(other)
    
    def __repr__(self) -> str:
        return f"TernaryArray(shape={self.shape}, sparsity={self.sparsity:.2%})"
    
    @classmethod
    def random(cls, shape: Tuple, sparsity: float = 0.5, 
               seed: int = None) -> 'TernaryArray':
        """
        Create random ternary array with specified sparsity.
        
        Args:
            shape: Shape of the array
            sparsity: Target fraction of zeros (0 to 1)
            seed: Random seed for reproducibility
        
        Returns:
            Random TernaryArray
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random values
        data = np.random.choice(
            [-1, 0, 1],
            size=shape,
            p=[(1-sparsity)/2, sparsity, (1-sparsity)/2]
        )
        return cls(data, copy=False)
    
    @classmethod
    def from_continuous(cls, data: np.ndarray, 
                        threshold: float = 0.33) -> 'TernaryArray':
        """
        Create ternary array by quantizing continuous values.
        
        Args:
            data: Continuous input data
            threshold: Threshold for non-zero assignment
        
        Returns:
            Quantized TernaryArray
        """
        quantized = np.where(data > threshold, 1,
                            np.where(data < -threshold, -1, 0))
        return cls(quantized, copy=False)


class QuinaryArray:
    """
    A specialized array class for quinary values {-2, -1, 0, +1, +2}.
    
    Used for neuron activations to provide more expressiveness than ternary
    while remaining computationally efficient.
    """
    
    def __init__(self, data: Union[np.ndarray, List, Tuple], copy: bool = True):
        """
        Initialize quinary array.
        
        Args:
            data: Input data (will be quantized to quinary)
            copy: Whether to copy the data
        """
        if isinstance(data, np.ndarray):
            arr = data.copy() if copy else data
        else:
            arr = np.array(data)
        
        # Quantize to quinary
        self._data = np.clip(np.round(arr), -2, 2).astype(np.int8)
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @property
    def shape(self) -> Tuple:
        return self._data.shape
    
    @classmethod
    def from_continuous(cls, data: np.ndarray) -> 'QuinaryArray':
        """
        Create quinary array by quantizing continuous values.
        
        Uses thresholds: [-1.5, -0.5, 0.5, 1.5] to map to {-2, -1, 0, +1, +2}
        
        Args:
            data: Continuous input data
        
        Returns:
            Quantized QuinaryArray
        """
        quantized = np.zeros_like(data, dtype=np.int8)
        quantized[data <= -1.5] = -2
        quantized[(data > -1.5) & (data <= -0.5)] = -1
        quantized[(data > -0.5) & (data <= 0.5)] = 0
        quantized[(data > 0.5) & (data <= 1.5)] = 1
        quantized[data > 1.5] = 2
        return cls(quantized, copy=False)
    
    def __repr__(self) -> str:
        return f"QuinaryArray(shape={self.shape})"


# Balanced ternary representation utilities

def int_to_balanced_ternary(n: int, min_digits: int = 1) -> List[int]:
    """
    Convert integer to balanced ternary representation.
    
    Args:
        n: Integer to convert
        min_digits: Minimum number of digits in output
    
    Returns:
        List of trits (least significant first)
    
    Example:
        >>> int_to_balanced_ternary(8)
        [-1, 0, 1]  # 8 = 9 - 1 = 3^2 - 3^0
    """
    if n == 0:
        return [0] * min_digits
    
    trits = []
    while n != 0:
        remainder = n % 3
        if remainder == 0:
            trits.append(0)
            n = n // 3
        elif remainder == 1:
            trits.append(1)
            n = n // 3
        else:  # remainder == 2
            trits.append(-1)
            n = (n + 1) // 3
    
    # Pad to minimum digits
    while len(trits) < min_digits:
        trits.append(0)
    
    return trits


def balanced_ternary_to_int(trits: List[int]) -> int:
    """
    Convert balanced ternary representation to integer.
    
    Args:
        trits: List of trits (least significant first)
    
    Returns:
        Integer value
    
    Example:
        >>> balanced_ternary_to_int([-1, 0, 1])
        8  # -1*1 + 0*3 + 1*9 = 8
    """
    result = 0
    for i, trit in enumerate(trits):
        result += trit * (3 ** i)
    return result


def ternary_add(a: List[int], b: List[int]) -> List[int]:
    """
    Add two balanced ternary numbers.
    
    Args:
        a: First number (list of trits, LSB first)
        b: Second number (list of trits, LSB first)
    
    Returns:
        Sum (list of trits, LSB first)
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a = a + [0] * (max_len - len(a))
    b = b + [0] * (max_len - len(b))
    
    result = []
    carry = 0
    
    for i in range(max_len):
        total = a[i] + b[i] + carry
        
        if total >= 2:
            result.append(total - 3)
            carry = 1
        elif total <= -2:
            result.append(total + 3)
            carry = -1
        else:
            result.append(total)
            carry = 0
    
    if carry != 0:
        result.append(carry)
    
    # Remove trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    return result


# Activation functions for ternary/quinary networks

def ternary_activation(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Ternary activation function.
    
    Maps continuous values to {-1, 0, +1} based on thresholds.
    
    Args:
        x: Input values
        threshold: Threshold for non-zero output
    
    Returns:
        Ternary activations
    """
    return np.where(x > threshold, 1,
                   np.where(x < -threshold, -1, 0)).astype(np.int8)


def quinary_activation(x: np.ndarray) -> np.ndarray:
    """
    Quinary activation function.
    
    Maps continuous values to {-2, -1, 0, +1, +2}.
    
    Args:
        x: Input values
    
    Returns:
        Quinary activations
    """
    result = np.zeros_like(x, dtype=np.int8)
    result[x <= -1.5] = -2
    result[(x > -1.5) & (x <= -0.5)] = -1
    result[(x > 0.5) & (x <= 1.5)] = 1
    result[x > 1.5] = 2
    return result


def soft_ternary(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Soft ternary activation using tanh-like function.
    
    Provides differentiable approximation to ternary quantization.
    
    Args:
        x: Input values
        temperature: Controls sharpness (lower = sharper)
    
    Returns:
        Soft ternary values in range [-1, 1]
    """
    return np.tanh(x / temperature)


def straight_through_ternary(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Straight-through estimator for ternary quantization.
    
    Returns both the quantized forward pass and the gradient-friendly
    backward pass (identity gradient within [-1, 1]).
    
    Args:
        x: Input values
    
    Returns:
        Tuple of (quantized values, gradient mask)
    """
    quantized = ternary_activation(x)
    # Gradient flows through where |x| <= 1
    grad_mask = (np.abs(x) <= 1.0).astype(np.float32)
    return quantized, grad_mask


if __name__ == "__main__":
    # Test ternary operations
    print("=== Balanced Ternary Tests ===\n")
    
    # Test integer conversion
    for n in [0, 1, -1, 8, -9, 29, -29]:
        trits = int_to_balanced_ternary(n)
        recovered = balanced_ternary_to_int(trits)
        print(f"{n:4d} -> {trits} -> {recovered}")
    
    print("\n=== Ternary Array Tests ===\n")
    
    # Test TernaryArray
    ta = TernaryArray.random((5, 5), sparsity=0.5, seed=42)
    print(f"Random ternary array: {ta}")
    print(f"Data:\n{ta.data}\n")
    
    # Test dot product
    x = np.random.randn(5)
    result = ta.dot(x)
    print(f"Input: {x}")
    print(f"Dot product: {result}\n")
    
    # Test quantization
    continuous = np.random.randn(10) * 2
    ta_quant = TernaryArray.from_continuous(continuous, threshold=0.5)
    print(f"Continuous: {continuous}")
    print(f"Quantized:  {ta_quant.data}\n")
    
    print("=== Quinary Activation Tests ===\n")
    
    x = np.array([-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3])
    qa = quinary_activation(x)
    print(f"Input:    {x}")
    print(f"Quinary:  {qa}")
