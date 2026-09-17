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

/// Returns a flattened 1D view when the array is contiguous, otherwise a flattened copy.
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
/// Convert flat linear indices to an [n, ndim] coordinate array (dtype .i64).
pub fn unravelIndex(
    allocator: std.mem.Allocator,
    flat_indices: []const usize,
    dims: []const usize,
) !Array;

/// Convert an [n, ndim] coordinate array to flat linear indices.
pub fn ravelIndex(
    allocator: std.mem.Allocator,
    coords: Array,
    dims: []const usize,
) !Array;

/// Generate a single Array of grid coordinates of shape [ndim, ...dims] (dtype .i64).
pub fn indices(allocator: std.mem.Allocator, dims: []const usize) !Array;
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

/// Stack 1D arrays in sequence (axis 0) or join higher-rank arrays along axis 1.
pub fn hstack(arrays: []const Array) !Array;

/// Promote 1D inputs to rows with `atleast2d` and join along axis 0.
pub fn vstack(arrays: []const Array) !Array;

/// Split an array into equal parts along an axis.
pub fn split(
    allocator: std.mem.Allocator,
    arr: Array,
    options: struct { parts: usize, axis: isize = 0 },
) ![]Array;

/// Append values to an array (flattened when axis is null, else along the axis).
pub fn append(arr: Array, values: Array, options: struct { axis: ?isize = null }) !Array;

/// Insert values at an index along an axis (flattened when axis is null).
pub fn insert(arr: Array, index: usize, values: Array, options: struct { axis: ?isize = null }) !Array;

/// Delete the entry at an index along an axis (flattened when axis is null).
pub fn delete(arr: Array, index: usize, options: struct { axis: ?isize = null }) !Array;
```

---

## Tiling, Repeating & Padding

```zig
/// Construct array by repeating `arr` per-dimension according to `reps`.
pub fn tile(arr: Array, options: struct { reps: []const usize }) !Array;

/// Repeat elements of an array.
pub fn repeat(arr: Array, options: struct { repeats: usize, axis: ?isize = null }) !Array;

/// Pad array per `pad_width[i] = [before, after]` with `mode` (.constant, .edge, .reflect).
pub fn pad(arr: Array, options: anytype) !Array;
pub const PadMode = enum { constant, edge, reflect };
```

