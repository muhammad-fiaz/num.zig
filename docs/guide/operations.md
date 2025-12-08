# Operations

`num.zig` provides a comprehensive set of element-wise mathematical operations that work efficiently on arrays of any shape.

## Arithmetic Operations

### Basic Arithmetic

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var a = try num.NDArray(f64).arange(allocator, 1.0, 6.0, 1.0);
    defer a.deinit(allocator);
    
    var b = try num.NDArray(f64).full(allocator, &.{5}, 2.0);
    defer b.deinit(allocator);
    
    try a.print();
    try b.print();
}
```

**Output:**
```
a: [1.0, 2.0, 3.0, 4.0, 5.0]
b: [2.0, 2.0, 2.0, 2.0, 2.0]
```

#### Addition (`add`)

```zig
var result = try num.ops.add(allocator, f64, a, b);
defer result.deinit(allocator);
try result.print();
// Output: [3.0, 4.0, 5.0, 6.0, 7.0]
```

#### Subtraction (`sub`)

```zig
var result = try num.ops.sub(allocator, f64, a, b);
defer result.deinit(allocator);
try result.print();
// Output: [-1.0, 0.0, 1.0, 2.0, 3.0]
```

#### Multiplication (`mul`)

```zig
var result = try num.ops.mul(allocator, f64, a, b);
defer result.deinit(allocator);
try result.print();
// Output: [2.0, 4.0, 6.0, 8.0, 10.0]
```

#### Division (`div`)

```zig
var result = try num.ops.div(allocator, f64, a, b);
defer result.deinit(allocator);
try result.print();
// Output: [0.5, 1.0, 1.5, 2.0, 2.5]
```

## Power and Root Operations

### Power (`pow`)

```zig
var arr = try num.NDArray(f64).arange(allocator, 1.0, 6.0, 1.0);
defer arr.deinit(allocator);

var result = try num.ops.pow(allocator, f64, arr, 2.0);
defer result.deinit(allocator);
try result.print();
// Output: [1.0, 4.0, 9.0, 16.0, 25.0]

// Cube each element
var cubed = try num.ops.pow(allocator, f64, arr, 3.0);
defer cubed.deinit(allocator);
try cubed.print();
// Output: [1.0, 8.0, 27.0, 64.0, 125.0]
```

### Square Root (`sqrt`)

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 10.0, 1.0);
defer arr.deinit(allocator);

var result = try num.ops.sqrt(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 1.0, 1.414, 1.732, 2.0, 2.236, 2.449, 2.646, 2.828, 3.0]
```

## Exponential and Logarithmic

### Exponential (`exp`)

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 5.0, 1.0);
defer arr.deinit(allocator);

var result = try num.ops.exp(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [1.0, 2.718, 7.389, 20.086, 54.598]
```

### Natural Logarithm (`log`)

```zig
var arr = try num.NDArray(f64).arange(allocator, 1.0, 6.0, 1.0);
defer arr.deinit(allocator);

var result = try num.ops.log(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 0.693, 1.099, 1.386, 1.609]
```

### Logarithm Base 10 (`log10`)

```zig
var arr = try num.NDArray(f64).full(allocator, &.{5}, 100.0);
defer arr.deinit(allocator);
for (0..5) |i| arr.data[i] = std.math.pow(f64, 10.0, @floatFromInt(i));

var result = try num.ops.log10(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 1.0, 2.0, 3.0, 4.0]
```

### Logarithm Base 2 (`log2`)

```zig
var arr = try num.NDArray(f64).full(allocator, &.{5}, 1.0);
defer arr.deinit(allocator);
for (0..5) |i| arr.data[i] = std.math.pow(f64, 2.0, @floatFromInt(i));

var result = try num.ops.log2(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 1.0, 2.0, 3.0, 4.0]
```

## Trigonometric Functions

### Sine (`sin`)

```zig
const pi = std.math.pi;
var angles = try num.NDArray(f64).linspace(allocator, 0.0, 2.0 * pi, 5);
defer angles.deinit(allocator);

var result = try num.ops.sin(allocator, f64, angles);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 1.0, 0.0, -1.0, 0.0]
```

### Cosine (`cos`)

```zig
const pi = std.math.pi;
var angles = try num.NDArray(f64).linspace(allocator, 0.0, 2.0 * pi, 5);
defer angles.deinit(allocator);

var result = try num.ops.cos(allocator, f64, angles);
defer result.deinit(allocator);
try result.print();
// Output: [1.0, 0.0, -1.0, 0.0, 1.0]
```

### Tangent (`tan`)

```zig
const pi = std.math.pi;
var angles = try num.NDArray(f64).full(allocator, &.{4}, 0.0);
defer angles.deinit(allocator);
angles.data[0] = 0.0;
angles.data[1] = pi / 4.0;
angles.data[2] = pi / 2.0;
angles.data[3] = 3.0 * pi / 4.0;

var result = try num.ops.tan(allocator, f64, angles);
defer result.deinit(allocator);
try result.print();
// Output: [0.0, 1.0, ∞, -1.0]
```

### Inverse Trigonometric Functions

```zig
// arcsin
var arr = try num.NDArray(f64).linspace(allocator, -1.0, 1.0, 5);
defer arr.deinit(allocator);
var result = try num.ops.arcsin(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [-π/2, -π/4, 0.0, π/4, π/2]

// arccos
var result2 = try num.ops.arccos(allocator, f64, arr);
defer result2.deinit(allocator);
try result2.print();
// Output: [π, 3π/4, π/2, π/4, 0.0]

// arctan
var arr2 = try num.NDArray(f64).arange(allocator, -2.0, 3.0, 1.0);
defer arr2.deinit(allocator);
var result3 = try num.ops.arctan(allocator, f64, arr2);
defer result3.deinit(allocator);
try result3.print();
// Output: [-1.107, -0.785, 0.0, 0.785, 1.107]
```

## Hyperbolic Functions

### Sinh, Cosh, Tanh

```zig
var arr = try num.NDArray(f64).arange(allocator, -2.0, 3.0, 1.0);
defer arr.deinit(allocator);

// Hyperbolic sine
var sinh_result = try num.ops.sinh(allocator, f64, arr);
defer sinh_result.deinit(allocator);
try sinh_result.print();
// Output: [-3.627, -1.175, 0.0, 1.175, 3.627]

// Hyperbolic cosine
var cosh_result = try num.ops.cosh(allocator, f64, arr);
defer cosh_result.deinit(allocator);
try cosh_result.print();
// Output: [3.762, 1.543, 1.0, 1.543, 3.762]

// Hyperbolic tangent
var tanh_result = try num.ops.tanh(allocator, f64, arr);
defer tanh_result.deinit(allocator);
try tanh_result.print();
// Output: [-0.964, -0.762, 0.0, 0.762, 0.964]
```

## Rounding and Absolute Value

### Floor

```zig
var arr = try num.NDArray(f64).linspace(allocator, -2.5, 2.5, 6);
defer arr.deinit(allocator);
try arr.print();
// Input: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

var result = try num.ops.floor(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
```

### Ceiling

```zig
var arr = try num.NDArray(f64).linspace(allocator, -2.5, 2.5, 6);
defer arr.deinit(allocator);

var result = try num.ops.ceil(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
```

### Round

```zig
var arr = try num.NDArray(f64).linspace(allocator, -2.7, 2.7, 7);
defer arr.deinit(allocator);

var result = try num.ops.round(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
```

### Absolute Value

```zig
var arr = try num.NDArray(f64).arange(allocator, -5.0, 6.0, 2.0);
defer arr.deinit(allocator);
try arr.print();
// Input: [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]

var result = try num.ops.abs(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [5.0, 3.0, 1.0, 1.0, 3.0, 5.0]
```

## Sign and Clipping

### Sign

```zig
var arr = try num.NDArray(f64).arange(allocator, -3.0, 4.0, 1.0);
defer arr.deinit(allocator);
try arr.print();
// Input: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

var result = try num.ops.sign(allocator, f64, arr);
defer result.deinit(allocator);
try result.print();
// Output: [-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
```

### Clip

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 11.0, 1.0);
defer arr.deinit(allocator);

var result = try num.ops.clip(allocator, f64, arr, 2.0, 8.0);
defer result.deinit(allocator);
try result.print();
// Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Output: [2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8]
```

## Comparison Operations

### Maximum and Minimum (Element-wise)

```zig
var a = try num.NDArray(f64).arange(allocator, 0.0, 5.0, 1.0);
defer a.deinit(allocator);

var b = try num.NDArray(f64).full(allocator, &.{5}, 2.5);
defer b.deinit(allocator);

// Element-wise maximum
var max_result = try num.ops.maximum(allocator, f64, a, b);
defer max_result.deinit(allocator);
try max_result.print();
// a:      [0.0, 1.0, 2.0, 3.0, 4.0]
// b:      [2.5, 2.5, 2.5, 2.5, 2.5]
// Output: [2.5, 2.5, 2.5, 3.0, 4.0]

// Element-wise minimum
var min_result = try num.ops.minimum(allocator, f64, a, b);
defer min_result.deinit(allocator);
try min_result.print();
// Output: [0.0, 1.0, 2.0, 2.5, 2.5]
```

## Practical Example: Data Normalization

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Sample data
    var data = try num.NDArray(f64).arange(allocator, 0.0, 100.0, 10.0);
    defer data.deinit(allocator);
    try data.print();
    // Input: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    // Compute statistics
    const mean = try num.stats.mean(allocator, f64, data);
    const std_dev = try num.stats.stdDev(allocator, f64, data);
    
    std.debug.print("Mean: {d}, Std: {d}\n", .{mean, std_dev});
    // Output: Mean: 45.0, Std: 28.723
    
    // Normalize: (x - mean) / std
    var centered = try num.ops.sub_scalar(allocator, f64, data, mean);
    defer centered.deinit(allocator);
    
    var normalized = try num.ops.div_scalar(allocator, f64, centered, std_dev);
    defer normalized.deinit(allocator);
    
    try normalized.print();
    // Output: [-1.566, -1.218, -0.871, -0.523, -0.175, 
    //          0.175, 0.523, 0.871, 1.218, 1.566]
}
```

## Performance Considerations

All operations in `num.zig` are:

- **Vectorized**: Use SIMD instructions when available
- **Memory-efficient**: Work with views when possible
- **Type-safe**: Compile-time type checking
- **Cache-friendly**: Contiguous memory access patterns

```zig
// Efficient chaining
var result = try num.ops.exp(
    allocator,
    f64,
    try num.ops.mul(
        allocator,
        f64,
        a,
        b
    )
);
// Computes: exp(a * b)
```
