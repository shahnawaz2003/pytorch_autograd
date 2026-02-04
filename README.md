# PyTorch Autograd Practice

A comprehensive Jupyter notebook demonstrating PyTorch's automatic differentiation (autograd) system with practical examples.

## Overview

This notebook covers fundamental concepts of PyTorch's autograd mechanism, which is essential for training neural networks. It includes both simple examples and real-world scenarios like binary classification with gradient computation.

## Topics Covered

### 1. **Basic Autograd Mechanics**
- Using `requires_grad=True` to enable gradient tracking
- Computing gradients with `.backward()`
- Understanding the computation graph (`grad_fn`)
- Simple derivative calculations (e.g., y = x²)

### 2. **Chain Rule in Autograd**
- Multi-step computations with automatic differentiation
- Example: `z = sin(x²)` demonstrating chain rule application
- Gradient computation through multiple operations

### 3. **Manual vs Automatic Backpropagation**

#### **"Aam Zindagi" (Normal Life) - Manual Chain Rule**
Manual implementation of backpropagation for binary classification:
- Forward pass: `z = wx + b`, `y_pred = sigmoid(z)`
- Loss calculation: Binary Cross-Entropy
- Manual gradient computation using chain rule
- Step-by-step calculation of `dL/dw` and `dL/db`

#### **"Mentos Zindagi" (Life with PyTorch) - Automatic Gradients**
Same binary classification problem solved with PyTorch's autograd:
- Automatic gradient computation
- Comparison with manual calculations
- Demonstrates efficiency and accuracy of autograd

### 4. **Vector Operations**
- Autograd with multi-dimensional tensors
- Computing gradients for vectors
- Mean reduction and gradient flow

### 5. **Gradient Accumulation**
- Understanding gradient accumulation behavior
- Using `.zero_()` to reset gradients
- Why clearing gradients is important in training loops

### 6. **Disabling Gradient Tracking**
Three methods to disable gradient computation (important for inference):

**Option 1: `requires_grad_(False)`**
```python
x.requires_grad_(False)
```

**Option 2: `.detach()`**
```python
z = x.detach()
```

**Option 3: `torch.no_grad()` context manager**
```python
with torch.no_grad():
    y = x ** 2
```

## Key Concepts Demonstrated

- **Computation Graph**: PyTorch builds a dynamic computational graph tracking all operations
- **Backpropagation**: `.backward()` computes gradients automatically using reverse-mode autodiff
- **Gradient Accumulation**: Gradients add up by default, requiring explicit zeroing
- **Memory Efficiency**: Disabling gradients during inference saves memory and computation
- **Binary Cross-Entropy**: Implementation and gradient computation for classification tasks

## Code Examples

### Simple Gradient Calculation
```python
x = torch.tensor(3.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Output: tensor(6.) [derivative of x² at x=3 is 2*3=6]
```

### Binary Classification with Manual Gradients
```python
# Manual backpropagation
d_loss_dy_pred = (y_pred - y) / (y_pred * (1 - y_pred))
dy_pred_d_z = y_pred * (1 - y_pred)
dz_d_w = x
dL_dw = d_loss_dy_pred * dy_pred_d_z * dz_d_w
```

### Binary Classification with Autograd
```python
# Automatic backpropagation
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
z = w*x + b
y_pred = torch.sigmoid(z)
loss = binary_cross_entropy(y_pred, y)
loss.backward()
print(w.grad, b.grad)
```

## Practical Applications

- **Neural Network Training**: Foundation for understanding how PyTorch trains models
- **Custom Loss Functions**: Creating and differentiating custom loss functions
- **Gradient-Based Optimization**: Understanding how optimizers update parameters
- **Debugging**: Using `.grad` to inspect gradient flow in your models

## Requirements

```python
import torch
```

## Learning Outcomes

After working through this notebook, you will understand:
- How PyTorch tracks operations for automatic differentiation
- The relationship between forward and backward passes
- When and how to disable gradient tracking
- The importance of gradient management in training loops
- How manual and automatic gradient computation compare

## Notes

- The notebook uses humorous section headers ("Aam Zindagi" vs "Mentos Zindagi") to contrast manual vs automatic approaches
- Error cases are intentionally shown to demonstrate gradient tracking requirements
- Both scalar and vector examples are provided for comprehensive understanding

## My Practice Focus

This notebook appears to be educational practice material focusing on:
- Building intuition for how autograd works internally
- Comparing manual calculations with PyTorch's automatic system
- Understanding common pitfalls (forgetting to zero gradients, using detached tensors)
- Preparing for more advanced deep learning implementations

---

*This is a practice notebook for learning PyTorch's automatic differentiation system.*
