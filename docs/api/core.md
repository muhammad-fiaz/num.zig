# Core API Reference
# Core API Reference

The `core` module provides the `NDArray` struct and fundamental array creation and manipulation functions.

## NDArray Struct

```zig
pub fn NDArray(comptime T: type) type
```

Generic N-dimensional array structure. The primary data structure in num.zig.

**Type Parameters:**
- `T`: Element type (e.g., `f32`, `i32`, `f64`)

**Fields:**
- `shape: []usize` - Dimensions of the array
- `strides: []usize` - Step sizes for each dimension
- `data: []T` - Flat data storage
- `offset: usize` - Starting offset in data array

---

## Array Creation

### zeros

```zig
pub fn zeros(allocator: Allocator, shape: []const usize) !NDArray(T)
```

Create array filled with zeros.

**Parameters:**
- `allocator`: Memory allocator
- `shape`: Array dimensions

**Returns:** NDArray filled with zeros

**Example:**
```zig
const allocator = std.heap.page_allocator;
const shape = [_]usize{2, 3};
var arr = try num.core.NDArray(f32).zeros(allocator, &shape);
defer arr.deinit(allocator);
// Result: [[0.0, 0.0, 0.0],
//          [0.0, 0.0, 0.0]]
```

---

### ones

```zig
pub fn ones(allocator: Allocator, shape: []const usize) !NDArray(T)
```

Create array filled with ones.

**Parameters:**
- `allocator`: Memory allocator
- `shape`: Array dimensions

**Returns:** NDArray filled with ones

**Example:**
```zig
const shape = [_]usize{3, 2};
var arr = try num.core.NDArray(f32).ones(allocator, &shape);
// Result: [[1.0, 1.0],
//          [1.0, 1.0],
//          [1.0, 1.0]]
```

---

### full

```zig
pub fn full(allocator: Allocator, shape: []const usize, value: T) !NDArray(T)
```

Create array filled with specified value.

**Parameters:**
- `allocator`: Memory allocator
- `shape`: Array dimensions
- `value`: Fill value

**Returns:** NDArray filled with `value`

**Example:**
```zig
const shape = [_]usize{2, 2};
var arr = try num.core.NDArray(f32).full(allocator, &shape, 7.5);
// Result: [[7.5, 7.5],
//          [7.5, 7.5]]
```

---

### eye

```zig
pub fn eye(allocator: Allocator, n: usize) !NDArray(T)
```

Create 2D identity matrix.

**Parameters:**
- `allocator`: Memory allocator
- `n`: Matrix size (n × n)

**Returns:** Identity matrix with ones on diagonal

**Example:**
```zig
var I = try num.core.NDArray(f32).eye(allocator, 3);
// Result: [[1.0, 0.0, 0.0],
//          [0.0, 1.0, 0.0],
//          [0.0, 0.0, 1.0]]
```

---

### arange

```zig
pub fn arange(allocator: Allocator, start: T, stop: T, step: T) !NDArray(T)
```

Create evenly spaced values within interval [start, stop).

**Parameters:**
- `allocator`: Memory allocator
- `start`: Start value (inclusive)
- `stop`: End value (exclusive)
- `step`: Spacing between values

**Returns:** 1D array of evenly spaced values

**Example:**
```zig
var arr = try num.core.NDArray(f32).arange(allocator, 0.0, 10.0, 2.0);
// Result: [0.0, 2.0, 4.0, 6.0, 8.0]
```

---

### linspace

```zig
pub fn linspace(allocator: Allocator, start: T, stop: T, num: usize) !NDArray(T)
```

Create `num` evenly spaced values over interval [start, stop].

**Parameters:**
- `allocator`: Memory allocator
- `start`: Start value (inclusive)
- `stop`: End value (inclusive)
- `num`: Number of samples

**Returns:** 1D array with `num` evenly spaced samples

**Example:**
```zig
var arr = try num.core.NDArray(f32).linspace(allocator, 0.0, 1.0, 5);
// Result: [0.0, 0.25, 0.5, 0.75, 1.0]
```

---

### init

```zig
pub fn init(shape: []const usize, data: []T) NDArray(T)
```

Initialize array with existing data.

**Parameters:**
- `shape`: Array dimensions
- `data`: Flat array data (must match shape)

**Returns:** NDArray wrapping provided data

**Example:**
```zig
const shape = [_]usize{2, 3};
var data = [_]f32{1, 2, 3, 4, 5, 6};
var arr = num.core.NDArray(f32).init(&shape, &data);
// Result: [[1.0, 2.0, 3.0],
//          [4.0, 5.0, 6.0]]
```

---

## Array Properties

### size

```zig
pub fn size(self: NDArray(T)) usize
```

Return total number of elements.

**Example:**
```zig
const shape = [_]usize{2, 3, 4};
var arr = try num.core.NDArray(f32).zeros(allocator, &shape);
const n = arr.size(); // Returns 24 (2 × 3 × 4)
```

---

### rank

```zig
pub fn rank(self: NDArray(T)) usize
```

Return number of dimensions.

**Example:**
```zig
const shape = [_]usize{2, 3, 4};
var arr = try num.core.NDArray(f32).zeros(allocator, &shape);
const r = arr.rank(); // Returns 3
```

---

## Array Access

### get

```zig
pub fn get(self: NDArray(T), indices: []const usize) !T
```

Get element at specified multi-dimensional index.

**Parameters:**
- `indices`: Array of indices (one per dimension)

**Returns:** Element value

**Example:**
```zig
const indices = [_]usize{1, 2};
const value = try arr.get(&indices);
```

---

### set

```zig
pub fn set(self: *NDArray(T), indices: []const usize, value: T) !void
```

Set element at specified multi-dimensional index.

**Parameters:**
- `indices`: Array of indices
- `value`: New value

**Example:**
```zig
const indices = [_]usize{0, 1};
try arr.set(&indices, 42.0);
```

---

### fill

```zig
pub fn fill(self: *NDArray(T), value: T) void
```

Fill entire array with scalar value.

**Parameters:**
- `value`: Fill value

**Example:**
```zig
arr.fill(3.14);
// All elements now equal 3.14
```

---

## Array Manipulation

### reshape

```zig
pub fn reshape(self: NDArray(T), new_shape: []const usize) !NDArray(T)
```

Return array with new shape (total size must match).

**Parameters:**
- `new_shape`: New dimensions

**Returns:** Reshaped view of array

**Example:**
```zig
// Original shape: (6,) = [1,2,3,4,5,6]
const new_shape = [_]usize{2, 3};
var reshaped = try arr.reshape(&new_shape);
// New shape: (2,3) = [[1,2,3], [4,5,6]]
```

---

### transpose

```zig
pub fn transpose(self: NDArray(T), allocator: Allocator) !NDArray(T)
```

Reverse dimensions (2D: swap rows and columns).

**Returns:** Transposed array

**Example:**
```zig
// Original: [[1, 2, 3],
//            [4, 5, 6]]
var T = try arr.transpose(allocator);
// Result:   [[1, 4],
//            [2, 5],
//            [3, 6]]
```

---

### copy

```zig
pub fn copy(self: NDArray(T), allocator: Allocator) !NDArray(T)
```

Create deep copy of array.

**Returns:** Independent copy with separate memory

**Example:**
```zig
var arr_copy = try arr.copy(allocator);
defer arr_copy.deinit(allocator);
// arr_copy is independent - modifying it doesn't affect arr
```

---

### concatenate

```zig
pub fn concatenate(allocator: Allocator, arrays: []const NDArray(T), axis: usize) !NDArray(T)
```

Join arrays along existing axis.

**Parameters:**
- `arrays`: Arrays to concatenate (must have compatible shapes)
- `axis`: Axis along which to join

**Returns:** Concatenated array

**Example:**
```zig
// arr1: [[1,2], [3,4]]  (2×2)
// arr2: [[5,6], [7,8]]  (2×2)
const arrays = [_]NDArray(f32){arr1, arr2};
var result = try num.core.concatenate(allocator, &arrays, 0);
// Result: [[1,2], [3,4], [5,6], [7,8]]  (4×2)
```

---

### stack

```zig
pub fn stack(allocator: Allocator, arrays: []const NDArray(T), axis: usize) !NDArray(T)
```

Join arrays along new axis.

**Parameters:**
- `arrays`: Arrays to stack (must have identical shapes)
- `axis`: New axis position

**Returns:** Stacked array with extra dimension

**Example:**
```zig
// arr1: [1,2,3]  (3,)
// arr2: [4,5,6]  (3,)
const arrays = [_]NDArray(f32){arr1, arr2};
var result = try num.core.stack(allocator, &arrays, 0);
// Result: [[1,2,3], [4,5,6]]  (2,3)
```

---

### expandDims

```zig
pub fn expandDims(self: NDArray(T), allocator: Allocator, axis: usize) !NDArray(T)
```

Expand shape by inserting new axis of length 1.

**Parameters:**
- `axis`: Position of new axis

**Returns:** Array with added dimension

**Example:**
```zig
// Original: [1,2,3]  shape=(3,)
var expanded = try arr.expandDims(allocator, 0);
// Result: [[1,2,3]]  shape=(1,3)
```

---

## Type Conversion

### astype

```zig
pub fn astype(self: NDArray(T), allocator: Allocator, comptime DestT: type) !NDArray(DestT)
```

Cast array elements to different type.

**Type Parameters:**
- `DestT`: Destination element type

**Returns:** New array with converted elements

**Example:**
```zig
var float_arr = try num.core.NDArray(f32).ones(allocator, &[_]usize{3});
var int_arr = try float_arr.astype(allocator, i32);
// Converts f32 -> i32
```

---

### asContiguous

```zig
pub fn asContiguous(self: NDArray(T), allocator: Allocator) !NDArray(T)
```

Return array with contiguous memory layout (C-order).

**Returns:** Contiguous array (may be copy if not already contiguous)

**Example:**
```zig
var contig = try arr.asContiguous(allocator);
// Guarantees data is stored in row-major order
```

---

## Utility Functions

### print

```zig
pub fn print(self: NDArray(T)) !void
```

Print array to stdout in formatted layout.

**Example:**
```zig
try arr.print();
// Output:
// [[1.0 2.0 3.0]
//  [4.0 5.0 6.0]]
```

---

### deinit

```zig
pub fn deinit(self: *NDArray(T), allocator: Allocator) void
```

Free array memory.

**Parameters:**
- `allocator`: Allocator used for creation

**Example:**
```zig
var arr = try num.core.NDArray(f32).zeros(allocator, &shape);
defer arr.deinit(allocator);
```

---

## See Also

- [Math Operations API](./math.md) - Element-wise mathematical operations
- [Indexing API](./indexing.md) - Advanced indexing and slicing
- [Manipulation API](./manipulation.md) - Shape manipulation functions