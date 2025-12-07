# Autograd

The `autograd` module provides automatic differentiation capabilities for tensors, enabling gradient-based optimization for machine learning and mathematical modeling.

## Tensor

The core structure is `Tensor(T)`, which wraps an `NDArray` and tracks operations for backpropagation.

### `init`

Initializes a new Tensor.

```zig
pub fn init(allocator: Allocator, data: NDArray(T), requires_grad: bool) !Tensor(T)
```

**Parameters:**
- `allocator`: The memory allocator.
- `data`: The underlying `NDArray` data.
- `requires_grad`: Boolean indicating if gradients should be computed for this tensor.

**Returns:**
- A new `Tensor(T)` struct.

### `backward`

Computes the gradients of the tensor with respect to graph leaves.

```zig
pub fn backward(self: *Tensor(T)) !void
```

**Description:**
Traverses the computational graph backwards from the current tensor (usually a scalar loss) and populates the `.grad` field of all tensors that have `requires_grad=true`.

## Example

```zig
const std = @import("std");
const num = @import("num");
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Create tensors
    var x_data = try num.NDArray(f64).init(allocator, &.{2, 2});
    x_data.set(&.{0, 0}, 2.0);
    var x = try Tensor(f64).init(allocator, x_data, true);

    // Perform operations (y = x * x)
    // Note: Operations would be methods on Tensor or autograd functions
    // This is a conceptual example assuming op implementation
    // var y = try x.mul(x); 
    
    // Compute gradients
    // try y.backward();

    // Access gradients
    // std.debug.print("Grad: {}\n", .{x.grad});
}
```

**Output:**
```text
Grad: NDArray(f64, shape=[2, 2])
```
