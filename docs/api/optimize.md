# Optimization API Reference

The `optimize` module provides optimization algorithms.

## Optimizers

### GradientDescent

Stochastic Gradient Descent optimizer.

```zig
pub fn GradientDescent(comptime T: type) type
```

- `init(lr: T)`: Initialize with learning rate.
- `step(param, grad)`: Update parameter using gradient.

### Adam

Adam optimizer.

```zig
pub fn Adam(comptime T: type) type
```

- `init(lr, beta1, beta2, epsilon)`: Initialize.
- `deinit(allocator)`: Deinitialize.
- `step(allocator, param, grad)`: Update parameter.

### RMSProp

RMSProp optimizer.

```zig
pub fn RMSProp(comptime T: type) type
```

- `init(lr, decay_rate, epsilon)`: Initialize.
- `deinit(allocator)`: Deinitialize.
- `step(allocator, param, grad)`: Update parameter.
