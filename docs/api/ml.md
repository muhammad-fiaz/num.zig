# Machine Learning

The `ml` module provides building blocks for neural networks.

## Activations

### `relu`

Computes the Rectified Linear Unit (ReLU) activation.

```zig
pub fn relu(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32)
```

### `sigmoid`

Computes the Sigmoid activation.

```zig
pub fn sigmoid(allocator: Allocator, a: *const NDArray(f32)) !NDArray(f32)
```

## Initialization Methods

The `InitMethod` enum defines strategies for initializing neural network weights.

```zig
pub const InitMethod = enum {
    RandomUniform,
    XavierUniform,
    HeNormal,
};
```

### `RandomUniform`

Initializes weights with a uniform distribution in the range `[-0.01, 0.01]`.

### `XavierUniform`

Initializes weights with a uniform distribution within a limit calculated as `sqrt(6 / (in + out))`. This is often used with Sigmoid or Tanh activation functions.

### `HeNormal`

Initializes weights with a normal distribution with a standard deviation of `sqrt(2 / in)`. This is often used with ReLU activation functions.

## Layers

### `Dense`

A fully connected layer.

#### `init`

```zig
pub fn init(allocator: Allocator, input_dim: usize, output_dim: usize, init_method: InitMethod) !Dense
```

#### `forward`

```zig
pub fn forward(self: *Dense, allocator: Allocator, input: *const NDArray(f32)) !NDArray(f32)
```

## Loss Functions

### `mse`

Mean Squared Error.

```zig
pub fn mse(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32
```

### `categoricalCrossEntropy`

Categorical Cross-Entropy loss.

```zig
pub fn categoricalCrossEntropy(allocator: Allocator, y_true: *const NDArray(f32), y_pred: *const NDArray(f32)) !f32
```

## Optimizers

### `SGD`

Stochastic Gradient Descent.

```zig
pub fn init(lr: f32) SGD
pub fn update(self: SGD, param: *NDArray(f32), grad: *const NDArray(f32)) void
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const Dense = num.ml.layers.Dense;
const SGD = num.ml.optim.SGD;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Layer
    var layer = try Dense.init(allocator, 10, 5, .XavierUniform);
    defer layer.deinit();

    // Optimizer
    var opt = SGD.init(0.01);

    // Forward pass (dummy input)
    var input = try num.NDArray(f32).zeros(allocator, &.{1, 10});
    var output = try layer.forward(allocator, &input);
    defer output.deinit();
}
```
