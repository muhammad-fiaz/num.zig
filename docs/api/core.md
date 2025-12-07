# Core API

The `core` module contains the fundamental `NDArray` structure and error definitions.

## NDArray(T)

N-dimensional array structure.

### `init`

Initializes a new NDArray.

```zig
pub fn init(allocator: Allocator, shape: []const usize) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `shape`: Slice representing dimensions.

**Returns:**
- A new `NDArray` instance.

### `zeros`

Creates an array filled with zeros.

```zig
pub fn zeros(allocator: Allocator, shape: []const usize) !NDArray(T)
```

### `ones`

Creates an array filled with ones.

```zig
pub fn ones(allocator: Allocator, shape: []const usize) !NDArray(T)
```

### `full`

Creates an array filled with a specific value.

```zig
pub fn full(allocator: Allocator, shape: []const usize, value: T) !NDArray(T)
```

### `arange`

Creates an array with evenly spaced values within a given interval.

```zig
pub fn arange(allocator: Allocator, start: T, stop: T, step: T) !NDArray(T)
```

### `linspace`

Creates an array with evenly spaced values over a specified interval.

```zig
pub fn linspace(allocator: Allocator, start: T, stop: T, num: usize) !NDArray(T)
```

### `eye`

Creates a 2D identity matrix.

```zig
pub fn eye(allocator: Allocator, n: usize) !NDArray(T)
```

### `flags`

Returns memory layout flags.

```zig
pub fn flags(self: Self) Flags
```

**Returns:**
- `Flags` struct (`c_contiguous`, `f_contiguous`, etc.).

## Error

Common errors used throughout the library.

```zig
pub const Error = error{
    ShapeMismatch,
    RankMismatch,
    AllocationFailed,
    IndexOutOfBounds,
    UnsupportedType,
    DimensionMismatch,
    NotImplemented,
    SingularMatrix,
    InvalidFormat,
    UnsupportedVersion,
};
```

