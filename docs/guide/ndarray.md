# NDArray Guide

The `NDArray` struct is the core of NumZig. It represents a multi-dimensional, homogeneous array of fixed-size items.

## Creation

You can create arrays in several ways:

- `init(allocator, shape)`: Uninitialized array.
- `zeros(allocator, shape)`: Filled with zeros.
- `ones(allocator, shape)`: Filled with ones.
- `full(allocator, shape, value)`: Filled with a specific value.
- `arange(allocator, start, stop, step)`: Range of values.
- `linspace(allocator, start, stop, num)`: Linearly spaced values.

## Memory Management

`NDArray` owns its data by default. You must call `deinit()` to free the memory.

```zig
var a = try NDArray(f32).zeros(allocator, &.{10, 10});
defer a.deinit();
```

## Views vs Copies

Some operations return a **view** of the data, meaning they share the underlying memory buffer. Modifying the view modifies the original array.

- `reshape`: Returns a view if possible.
- `transpose`: Returns a view with modified strides.
- `broadcastTo`: Returns a view with 0-strides for broadcasted dimensions.

Operations like `copy`, `flatten` (usually), and math operations return new arrays with their own memory.

## Accessing Elements

Use `get` and `set` with a slice of indices.

```zig
try a.set(&.{0, 1}, 5.0);
const val = try a.get(&.{0, 1});
```

## Slicing

You can extract sub-arrays using `num.indexing.slice`.

```zig
const num = @import("num");
// ...
// Slice equivalent to a[1:5:2]
var s = try num.indexing.slice(allocator, f32, a, &.{
    .{ .start = 1, .stop = 5, .step = 2 }
});
```

You can also use `num.indexing.take` to select specific elements along an axis.

