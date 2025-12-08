# Broadcasting

Broadcasting is a powerful mechanism that allows `num.zig` to perform operations on arrays of different shapes without explicitly copying data. This makes code more concise and efficient.

## Broadcasting Rules

Broadcasting follows these rules:

1. **Dimensions are compared from right to left** (trailing dimensions first)
2. **Two dimensions are compatible if:**
   - They are equal, or
   - One of them is 1

If all dimensions are compatible, the smaller array is "broadcast" across the larger array.

## Broadcast Utilities in num.zig

- `broadcastShape(shape_a, shape_b)`: Compute resulting shape for two arrays
- `broadcastShapes([][]usize)`: Compute a common shape for multiple inputs
- `NDArray.broadcastTo(new_shape)`: Create a broadcasted view without copying data

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape_a = [_]usize{3, 1};
    const shape_b = [_]usize{1, 4};

    const result_shape = try num.broadcast.broadcastShape(allocator, &shape_a, &shape_b);
    defer allocator.free(result_shape);

    std.debug.print("Broadcast shape of (3,1) and (1,4): {any}\n", .{result_shape});
}
// Output:
// Broadcast shape of (3,1) and (1,4): [3, 4]
```

## Visual Examples

### Example 1: Scalar Broadcasting

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Array shape: (3, 3)
    var arr = try num.NDArray(f64).ones(allocator, &.{3, 3});
    defer arr.deinit(allocator);
    
    // Multiply every element by 5
    // Scalar 5 is broadcast to shape (3, 3)
    const scalar: f64 = 5.0;
    for (0..arr.data.len) |i| {
        arr.data[i] *= scalar;
    }
    
    try arr.print();
}
```

**Output:**
```
[[5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0],
 [5.0, 5.0, 5.0]]
```

**Visual:**
```
    [1, 1, 1]        5      [5, 5, 5]
    [1, 1, 1]   ×         = [5, 5, 5]
    [1, 1, 1]              [5, 5, 5]
   (3, 3)      (scalar)    (3, 3)
```

### Example 2: 1D Array Broadcasting

```zig
// Array A: shape (3, 4)
var A = try num.NDArray(f64).ones(allocator, &.{3, 4});
defer A.deinit(allocator);
for (0..A.data.len) |i| {
    A.data[i] = @floatFromInt(i);
}

// Array B: shape (4,) - 1D array
var B = try num.NDArray(f64).arange(allocator, 1.0, 5.0, 1.0);
defer B.deinit(allocator);

// Broadcast B to match A
var B_b = try B.broadcastTo(allocator, &.{3, 4});
defer B_b.deinit(allocator);

// Elementwise add after broadcasting
var sum = try num.elementwise.add(allocator, f64, A, B_b);
defer sum.deinit(allocator);

try A.print();
try B.print();
try sum.print();
```

**Output:**
```
A (3, 4):
[[0, 1,  2,  3],
 [4, 5,  6,  7],
 [8, 9, 10, 11]]

B (4,):
[1, 2, 3, 4]

A + B (3, 4):
[[1,  3,  5,  7],
 [5,  7,  9, 11],
 [9, 11, 13, 15]]
```

**Visual:**
```
    [0,  1,  2,  3]       [1, 2, 3, 4]       [1,  3,  5,  7]
    [4,  5,  6,  7]   +   [1, 2, 3, 4]   =   [5,  7,  9, 11]
    [8,  9, 10, 11]       [1, 2, 3, 4]       [9, 11, 13, 15]
       (3, 4)                 (4,)               (3, 4)
                              ↓
                          broadcast
                         to (3, 4)
```

### Example 3: Column Vector Broadcasting

```zig
// Array A: shape (3, 4)
var A = try num.NDArray(f64).arange(allocator, 0.0, 12.0, 1.0);
try A.reshape(&.{3, 4});
defer A.deinit(allocator);

// Column vector: shape (3, 1)
var col = try num.NDArray(f64).ones(allocator, &.{3, 1});
col.data[0] = 10.0;
col.data[1] = 20.0;
col.data[2] = 30.0;
defer col.deinit(allocator);

try A.print();
try col.print();

// Broadcasting: col is broadcast to shape (3, 4)
```

**Output:**
```
A (3, 4):
[[0, 1,  2,  3],
 [4, 5,  6,  7],
 [8, 9, 10, 11]]

col (3, 1):
[[10],
 [20],
 [30]]

A + col (3, 4):
[[10, 11, 12, 13],
 [24, 25, 26, 27],
 [38, 39, 40, 41]]
```

**Visual:**
```
    [0, 1,  2,  3]       [10]       [10, 11, 12, 13]
    [4, 5,  6,  7]   +   [20]   =   [24, 25, 26, 27]
    [8, 9, 10, 11]       [30]       [38, 39, 40, 41]
       (3, 4)           (3, 1)          (3, 4)
                           ↓
                       broadcast
                      to (3, 4)
```

### Example 4: Both Arrays Broadcast

```zig
// Array A: shape (3, 1)
var A = try num.NDArray(f64).ones(allocator, &.{3, 1});
A.data[0] = 1.0;
A.data[1] = 2.0;
A.data[2] = 3.0;
defer A.deinit(allocator);

// Array B: shape (1, 4)
var B = try num.NDArray(f64).ones(allocator, &.{1, 4});
B.data[0] = 10.0;
B.data[1] = 20.0;
B.data[2] = 30.0;
B.data[3] = 40.0;
defer B.deinit(allocator);

try A.print();
try B.print();

// Result: both broadcast to (3, 4)
```

**Output:**
```
A (3, 1):
[[1],
 [2],
 [3]]

B (1, 4):
[[10, 20, 30, 40]]

A * B (3, 4):
[[10, 20, 30, 40],
 [20, 40, 60, 80],
 [30, 60, 90, 120]]
```

**Visual:**
```
    [1]              [10, 20, 30, 40]       [10, 20, 30,  40]
    [2]      ×                          =   [20, 40, 60,  80]
    [3]                                     [30, 60, 90, 120]
   (3, 1)               (1, 4)                  (3, 4)
     ↓                    ↓
  broadcast            broadcast
 to (3, 4)            to (3, 4)
```

## Dimension Compatibility Table

| Shape A | Shape B | Result | Compatible? |
|---------|---------|--------|-------------|
| (3, 4) | (4,) | (3, 4) | ✓ Yes |
| (3, 4) | (3, 1) | (3, 4) | ✓ Yes |
| (3, 1) | (1, 4) | (3, 4) | ✓ Yes |
| (3, 4) | (3, 2) | Error | ✗ No |
| (5, 3, 4) | (3, 4) | (5, 3, 4) | ✓ Yes |
| (5, 3, 4) | (3, 1) | (5, 3, 4) | ✓ Yes |
| (5, 1, 4) | (3, 4) | (5, 3, 4) | ✓ Yes |

## Advanced Broadcasting Examples

### Example 5: 3D Broadcasting

```zig
// 3D array: shape (2, 3, 4)
var arr3d = try num.NDArray(f64).zeros(allocator, &.{2, 3, 4});
defer arr3d.deinit(allocator);
for (0..arr3d.data.len) |i| {
    arr3d.data[i] = @floatFromInt(i);
}

// 2D array: shape (3, 4)
var arr2d = try num.NDArray(f64).ones(allocator, &.{3, 4});
defer arr2d.deinit(allocator);

// Broadcasting: arr2d broadcasts to (2, 3, 4)
// The 2D array is repeated across the first dimension
```

**Shape Analysis:**
```
arr3d:  (2, 3, 4)
arr2d:     (3, 4)
         --------
Result: (2, 3, 4)  ✓ Compatible
```

### Example 6: Normalizing Data

A common use case - normalize each feature (column) independently:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Data matrix: 5 samples, 3 features
    var data = try num.NDArray(f64).arange(allocator, 0.0, 15.0, 1.0);
    // Reshape to (5, 3)
    defer data.deinit(allocator);
    
    // Compute mean for each column
    var means = try num.stats.meanAxis(allocator, f64, data, 0);
    defer means.deinit(allocator);
    // means has shape (3,)
    
    try data.print();
    try means.print();
    
    // Subtract mean from each column (broadcasting)
    // data (5, 3) - means (3,) = result (5, 3)
}
```

**Output:**
```
Data (5, 3):
[[0,  1,  2],
 [3,  4,  5],
 [6,  7,  8],
 [9, 10, 11],
 [12, 13, 14]]

Means (3,):
[6.0, 7.0, 8.0]

Normalized (5, 3):
[[-6, -6, -6],
 [-3, -3, -3],
 [ 0,  0,  0],
 [ 3,  3,  3],
 [ 6,  6,  6]]
```

## Broadcasting in Practice

### Memory Efficiency

Broadcasting **doesn't copy data** - it creates a view with adjusted strides. This makes operations memory-efficient:

```zig
// Without broadcasting - wasteful
var large = try num.NDArray(f64).ones(allocator, &.{1000, 1000});
var repeated = try num.NDArray(f64).ones(allocator, &.{1000, 1000});
// Uses 2 × 1M × 8 bytes = 16 MB

// With broadcasting - efficient
var large = try num.NDArray(f64).ones(allocator, &.{1000, 1000});
var small = try num.NDArray(f64).ones(allocator, &.{1000});
// Uses (1M + 1K) × 8 bytes ≈ 8 MB
```

### Common Patterns

```zig
// Pattern 1: Add row vector to all rows
// (m, n) + (n,) → (m, n)

// Pattern 2: Add column vector to all columns
// (m, n) + (m, 1) → (m, n)

// Pattern 3: Outer product
// (m, 1) × (1, n) → (m, n)

// Pattern 4: Batch operations
// (batch, features) + (features,) → (batch, features)
```
