# Optimization

The `optimize` module provides algorithms for function optimization.

## GradientDescent

A simple gradient descent optimizer.

### `init`

Initializes the optimizer.

```zig
pub fn init(lr: T) Self
```

**Parameters:**
- `lr`: Learning rate (step size).

### `step`

Performs a single optimization step.

```zig
pub fn step(self: Self, param: *NDArray(T), grad: NDArray(T)) !void
```

**Parameters:**
- `param`: The parameters to update (in-place).
- `grad`: The gradient of the loss with respect to the parameters.

**Description:**
Updates parameters using the rule: `param = param - lr * grad`.

## Example

```zig
const std = @import("std");
const num = @import("num");
const GradientDescent = num.optimize.GradientDescent;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Initialize optimizer
    const optimizer = GradientDescent(f64).init(0.01);

    // Parameters and Gradient
    var param = try num.NDArray(f64).init(allocator, &.{2});
    param.set(&.{0}, 1.0);
    var grad = try num.NDArray(f64).init(allocator, &.{2});
    grad.set(&.{0}, 0.5);

    // Update
    try optimizer.step(&param, grad);
    
    // param[0] should be 1.0 - 0.01 * 0.5 = 0.995
}
```
