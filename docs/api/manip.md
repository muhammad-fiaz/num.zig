# Shape Manipulation API

Module: `@import("num").manip`

---

## Reshaping & Views

All reshape operations take an `Array` by value. `reshape` and `squeeze` return zero-allocation views when possible.

```zig
/// Reshapes to a new shape. Use -1 for one inferred dimension.
pub fn reshape(
    arr: Array,
    options: struct { shape: []const isize, order: Order = .c },
) !Array;

/// Returns a flattened 1D copy in C order.
pub fn ravel(arr: Array) !Array;

/// Returns a contiguous 1D copy, always allocating.
pub fn flatten(arr: Array) !Array;

/// Removes all size-1 dimensions (or a specific axis).
pub fn squeeze(arr: Array, options: struct { axis: ?isize = null }) !Array;

/// Inserts a new axis at the specified position.
pub fn expandDims(arr: Array, options: struct { axis: isize }) !Array;
```

---

## Rank Coercion

```zig
pub fn atleast1d(arr: Array) !Array;
pub fn atleast2d(arr: Array) !Array;
pub fn atleast3d(arr: Array) !Array;
```

---

## Index Utilities

```zig
/// Convert a flat linear index to N-dimensional indices.
pub fn unravelIndex(
    index: usize,
    shape: []const usize,
    options: struct { order: Order = .c },
) ![]usize;

/// Convert N-dimensional indices to a flat linear index.
pub fn ravelIndex(indices: []const usize, shape: []const usize, options: struct { order: Order = .c }) usize;

/// Generate an open mesh of indices for a given shape.
pub fn indices(allocator: std.mem.Allocator, shape: []const usize) ![]Array;
```

---

## Transposing & Axis Operations

Zero-allocation views unless a copy is required.

```zig
/// Reverse axes or apply a custom permutation.
pub fn transpose(arr: Array, options: struct { axes: ?[]const usize = null }) !Array;

/// Interchange two axes.
pub fn swapAxes(arr: Array, axis1: isize, axis2: isize) !Array;

/// Move a single axis to a new position.
pub fn moveAxis(arr: Array, source: isize, destination: isize) !Array;

/// Flip elements along an axis (or all axes).
pub fn flip(arr: Array, options: struct { axis: ?isize = null }) !Array;

/// Roll elements along an axis.
pub fn roll(arr: Array, options: struct { shift: isize, axis: ?isize = null }) !Array;
```

---

## Slicing

```zig
pub fn slice(arr: Array, slices: []const Slice) !Array;
```

---

## Join & Split

```zig
/// Concatenate arrays along an existing axis.
pub fn concat(arrays: []const Array, options: struct { axis: isize = 0 }) !Array;

/// Stack arrays along a new axis.
pub fn stack(arrays: []const Array, options: struct { axis: isize = 0 }) !Array;

/// Split an array into equal parts along an axis.
pub fn split(
    allocator: std.mem.Allocator,
    arr: Array,
    options: struct { parts: usize, axis: isize = 0 },
) ![]Array;
```

---

## Tiling, Repeating & Padding

```zig
/// Construct array by repeating `arr` per-dimension according to `reps`.
pub fn tile(arr: Array, options: struct { reps: []const usize }) !Array;

/// Repeat elements of an array.
pub fn repeat(arr: Array, options: struct { repeats: usize, axis: ?isize = null }) !Array;

/// Pad array with a constant value. `pad_width[i] = [before, after]` per axis.
pub fn pad(arr: Array, options: anytype) !Array;
```

