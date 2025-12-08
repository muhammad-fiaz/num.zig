# Autograd API Reference

The `autograd` module provides automatic differentiation.

## Tensor

The core data structure for autograd.

```zig
pub fn Tensor(comptime T: type) type
```

### Lifecycle

#### init

Initialize a tensor.

```zig
pub fn init(allocator: Allocator, data: NDArray(T), requires_grad: bool) !*Self
```

#### deinit

Deinitialize the tensor.

```zig
pub fn deinit(self: *Self, allocator: Allocator) void
```

### Gradients

#### backward

Compute gradients.

```zig
pub fn backward(self: *Self, allocator: Allocator) !void
```

### Operations

#### add

Element-wise addition.

```zig
pub fn add(self: *Self, allocator: Allocator, other: *Self) !*Self
```

#### mul

Element-wise multiplication.

```zig
pub fn mul(self: *Self, allocator: Allocator, other: *Self) !*Self
```

#### matmul

Matrix multiplication.

```zig
pub fn matmul(self: *Self, allocator: Allocator, other: *Self) !*Self
```

### Activation Functions

#### relu

Rectified Linear Unit.

```zig
pub fn relu(self: *Self, allocator: Allocator) !*Self
```

#### sigmoid

Sigmoid activation.

```zig
pub fn sigmoid(self: *Self, allocator: Allocator) !*Self
```

#### tanh

Hyperbolic tangent activation.

```zig
pub fn tanh(self: *Self, allocator: Allocator) !*Self
```

#### softmax

Softmax activation.

```zig
pub fn softmax(self: *Self, allocator: Allocator) !*Self
```

### Loss Functions

#### mse_loss

Mean Squared Error loss.

```zig
pub fn mse_loss(self: *Self, allocator: Allocator, target: *Self) !*Self
```

#### cross_entropy_loss

Cross Entropy loss.

```zig
pub fn cross_entropy_loss(self: *Self, allocator: Allocator, target: *Self) !*Self
```