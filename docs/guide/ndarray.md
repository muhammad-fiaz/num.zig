# N-Dimensional Arrays (NDArray)

The `NDArray` struct is the core data structure in `num.zig`. It represents a multidimensional, homogeneous array of fixed-size items, similar to NumPy's ndarray or MATLAB's matrix.

## What is an NDArray?

An NDArray is characterized by:

- **Shape**: The dimensions of the array (e.g., `[3, 4]` for a 3×4 matrix)
- **Data type**: The type of elements (`f32`, `f64`, `i32`, etc.)
- **Strides**: Internal layout for memory efficiency
- **Contiguity**: Memory layout (C-order by default)

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Create a 2×3 array
    var arr = try num.NDArray(f64).zeros(allocator, &.{2, 3});
    defer arr.deinit(allocator);
    
    std.debug.print("Shape: {any}\n", .{arr.shape});
    std.debug.print("Size: {}\n", .{arr.size()});
    std.debug.print("Rank: {}\n", .{arr.rank()});
}
```

**Output:**
```
Shape: [2, 3]
Size: 6
Rank: 2
```

## Array Creation

### Zeros and Ones

```zig
// 1D array of zeros
var arr1d = try num.NDArray(f64).zeros(allocator, &.{5});
defer arr1d.deinit(allocator);
try arr1d.print();
// Output: [0.0, 0.0, 0.0, 0.0, 0.0]

// 2D array of ones
var arr2d = try num.NDArray(f64).ones(allocator, &.{3, 4});
defer arr2d.deinit(allocator);
try arr2d.print();
// Output:
// [[1.0, 1.0, 1.0, 1.0],
//  [1.0, 1.0, 1.0, 1.0],
//  [1.0, 1.0, 1.0, 1.0]]

// 3D array of zeros
var arr3d = try num.NDArray(f64).zeros(allocator, &.{2, 3, 4});
defer arr3d.deinit(allocator);
std.debug.print("3D shape: {any}\n", .{arr3d.shape});
// Output: 3D shape: [2, 3, 4]
```

### Full (Constant Value)

```zig
var arr = try num.NDArray(f64).full(allocator, &.{2, 3}, 7.5);
defer arr.deinit(allocator);
try arr.print();
// Output:
// [[7.5, 7.5, 7.5],
//  [7.5, 7.5, 7.5]]
```

### Identity Matrix

```zig
var identity = try num.NDArray(f64).eye(allocator, 4);
defer identity.deinit(allocator);
try identity.print();
// Output:
// [[1.0, 0.0, 0.0, 0.0],
//  [0.0, 1.0, 0.0, 0.0],
//  [0.0, 0.0, 1.0, 0.0],
//  [0.0, 0.0, 0.0, 1.0]]
```

### Range Arrays

#### arange (like Python's range)

```zig
// Integer range
var int_range = try num.NDArray(i32).arange(allocator, 0, 10, 2);
defer int_range.deinit(allocator);
try int_range.print();
// Output: [0, 2, 4, 6, 8]

// Float range
var float_range = try num.NDArray(f64).arange(allocator, 0.0, 5.0, 0.5);
defer float_range.deinit(allocator);
try float_range.print();
// Output: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

// Negative step
var reverse = try num.NDArray(f64).arange(allocator, 10.0, 0.0, -2.0);
defer reverse.deinit(allocator);
try reverse.print();
// Output: [10.0, 8.0, 6.0, 4.0, 2.0]
```

#### linspace (evenly spaced values)

```zig
// 5 values from 0 to 1
var arr = try num.NDArray(f64).linspace(allocator, 0.0, 1.0, 5);
defer arr.deinit(allocator);
try arr.print();
// Output: [0.0, 0.25, 0.5, 0.75, 1.0]

// 10 values from -π to π
const pi = std.math.pi;
var angles = try num.NDArray(f64).linspace(allocator, -pi, pi, 10);
defer angles.deinit(allocator);
try angles.print();
// Output: [-3.14, -2.44, -1.75, -1.05, -0.35, 0.35, 1.05, 1.75, 2.44, 3.14]
```

## Accessing Elements

### Get and Set

```zig
var arr = try num.NDArray(f64).zeros(allocator, &.{3, 3});
defer arr.deinit(allocator);

// Set individual elements
try arr.set(&.{0, 0}, 1.0);
try arr.set(&.{1, 1}, 2.0);
try arr.set(&.{2, 2}, 3.0);

// Get elements
const val00 = try arr.get(&.{0, 0});
const val11 = try arr.get(&.{1, 1});
const val22 = try arr.get(&.{2, 2});

try arr.print();
// Output:
// [[1.0, 0.0, 0.0],
//  [0.0, 2.0, 0.0],
//  [0.0, 0.0, 3.0]]

std.debug.print("Diagonal: {d}, {d}, {d}\n", .{val00, val11, val22});
// Output: Diagonal: 1.0, 2.0, 3.0
```

### Direct Data Access

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 6.0, 1.0);
defer arr.deinit(allocator);

// Access underlying data
for (arr.data, 0..) |*elem, i| {
    elem.* = @floatFromInt(i * 10);
}

try arr.print();
// Output: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
```

## Shape and Dimensions

### Shape Information

```zig
var arr = try num.NDArray(f64).zeros(allocator, &.{2, 3, 4});
defer arr.deinit(allocator);

std.debug.print("Shape: {any}\n", .{arr.shape});
std.debug.print("Rank (ndim): {}\n", .{arr.rank()});
std.debug.print("Size (total elements): {}\n", .{arr.size()});

// Output:
// Shape: [2, 3, 4]
// Rank (ndim): 3
// Size (total elements): 24
```

### Reshaping

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 12.0, 1.0);
defer arr.deinit(allocator);

std.debug.print("Original shape: {any}\n", .{arr.shape});
// Output: Original shape: [12]

// Reshape to 3×4
var reshaped = try arr.reshape(allocator, &.{3, 4});
defer reshaped.deinit(allocator);
try reshaped.print();
// Output:
// [[0.0,  1.0,  2.0,  3.0],
//  [4.0,  5.0,  6.0,  7.0],
//  [8.0,  9.0, 10.0, 11.0]]

// Reshape to 2×2×3
var reshaped3d = try arr.reshape(allocator, &.{2, 2, 3});
defer reshaped3d.deinit(allocator);
std.debug.print("New shape: {any}\n", .{reshaped3d.shape});
// Output: New shape: [2, 2, 3]
```

### Expanding Dimensions

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 5.0, 1.0);
defer arr.deinit(allocator);

std.debug.print("Original: {any}\n", .{arr.shape});
// Output: Original: [5]

// Add dimension at axis 0: [5] -> [1, 5]
var expanded = try arr.expandDims(allocator, 0);
defer expanded.deinit(allocator);
std.debug.print("Expanded: {any}\n", .{expanded.shape});
// Output: Expanded: [1, 5]
```

## Array Operations

### Copy

```zig
var original = try num.NDArray(f64).arange(allocator, 0.0, 5.0, 1.0);
defer original.deinit(allocator);

var copy = try original.copy(allocator);
defer copy.deinit(allocator);

// Modify copy
copy.data[0] = 99.0;

try original.print();
// Output: [0.0, 1.0, 2.0, 3.0, 4.0] (unchanged)

try copy.print();
// Output: [99.0, 1.0, 2.0, 3.0, 4.0]
```

### Fill

```zig
var arr = try num.NDArray(f64).zeros(allocator, &.{3, 3});
defer arr.deinit(allocator);

arr.fill(42.0);

try arr.print();
// Output:
// [[42.0, 42.0, 42.0],
//  [42.0, 42.0, 42.0],
//  [42.0, 42.0, 42.0]]
```

### Type Conversion

```zig
var float_arr = try num.NDArray(f64).linspace(allocator, 0.0, 10.0, 6);
defer float_arr.deinit(allocator);
try float_arr.print();
// Output: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

var int_arr = try float_arr.astype(allocator, i32);
defer int_arr.deinit(allocator);
try int_arr.print();
// Output: [0, 2, 4, 6, 8, 10]
```

## Concatenation

```zig
var a = try num.NDArray(f64).ones(allocator, &.{2, 3});
defer a.deinit(allocator);

var b = try num.NDArray(f64).zeros(allocator, &.{2, 3});
defer b.deinit(allocator);

// Concatenate along axis 0 (rows)
var arrays = [_]num.NDArray(f64){a, b};
var result = try num.NDArray(f64).concatenate(allocator, &arrays, 0);
defer result.deinit(allocator);

try result.print();
// Output:
// [[1.0, 1.0, 1.0],
//  [1.0, 1.0, 1.0],
//  [0.0, 0.0, 0.0],
//  [0.0, 0.0, 0.0]]

std.debug.print("Shape: {any}\n", .{result.shape});
// Output: Shape: [4, 3]
```

## Memory Management

### Contiguous Arrays

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 12.0, 1.0);
defer arr.deinit(allocator);

// Ensure array is contiguous in memory
var contiguous = try arr.asContiguous(allocator);
defer contiguous.deinit(allocator);

const flags = arr.flags();
std.debug.print("C-contiguous: {}\n", .{flags.c_contiguous});
std.debug.print("F-contiguous: {}\n", .{flags.f_contiguous});
```

### Cleanup

Always call `deinit` to free memory:

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var arr = try num.NDArray(f64).zeros(allocator, &.{1000, 1000});
    defer arr.deinit(allocator); // Important!
    
    // Use arr...
}
```

## Practical Example: Image Processing

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Simulate a 4×4 grayscale image
    var image = try num.NDArray(u8).zeros(allocator, &.{4, 4});
    defer image.deinit(allocator);
    
    // Fill with gradient
    for (0..4) |i| {
        for (0..4) |j| {
            const idx = i * 4 + j;
            image.data[idx] = @intCast((i + j) * 16);
        }
    }
    
    try image.print();
    // Output:
    // [[  0,  16,  32,  48],
    //  [ 16,  32,  48,  64],
    //  [ 32,  48,  64,  80],
    //  [ 48,  64,  80,  96]]
    
    // Convert to float for processing
    var float_image = try image.astype(allocator, f32);
    defer float_image.deinit(allocator);
    
    // Normalize to [0, 1]
    const max_val: f32 = 255.0;
    for (float_image.data) |*pixel| {
        pixel.* /= max_val;
    }
    
    try float_image.print();
    // Output: normalized values between 0.0 and 1.0
}
```

