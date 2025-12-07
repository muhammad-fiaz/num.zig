# Broadcasting

Broadcasting is a powerful mechanism that allows Num.Zig to work with arrays of different shapes during arithmetic operations. Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes.

## General Broadcasting Rules

When operating on two arrays, Num.Zig compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimensions and works its way left. Two dimensions are compatible when:

1. They are equal, or
2. One of them is 1

If these conditions are not met, a `ShapeMismatch` error is thrown.

## Examples

### Scalar and Array

A scalar is treated as a 0-D array, which broadcasts to any shape.

```zig
var a = try NDArray(f32).ones(allocator, &.{2, 3});
// a is:
// [[1, 1, 1],
//  [1, 1, 1]]

// Add scalar 2.0
// Effectively adds 2.0 to every element
var b = try num.ops.addScalar(f32, allocator, &a, 2.0); 
// b is:
// [[3, 3, 3],
//  [3, 3, 3]]
```

### 1D and 2D Array

```zig
// A: (2, 3)
// B: (3,)
// Result: (2, 3)

var a = try NDArray(f32).ones(allocator, &.{2, 3});
var b = try NDArray(f32).init(allocator, &.{3});
// fill b with [1, 2, 3]

var c = try num.ops.add(f32, allocator, &a, &b);
// b is broadcast across each row of a.
```

### Incompatible Shapes

```zig
// A: (2, 3)
// B: (2,)
// Trailing dimensions: 3 vs 2 -> Mismatch!
```

To fix this, you might need to reshape `B` to `(2, 1)`.

```zig
var b_reshaped = try b.reshape(allocator, &.{2, 1});
// Now:
// A: (2, 3)
// B: (2, 1)
// Trailing: 3 vs 1 -> OK (1 stretches to 3)
// Next:     2 vs 2 -> OK
// Result: (2, 3)
```
