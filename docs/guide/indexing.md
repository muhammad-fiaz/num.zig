# Indexing & Slicing

`num.zig` provides flexible indexing primitives for multi-dimensional scalar access, sub-tensor slicing, and strided views.

---

## 1. Scalar Indexing (`get` and `set`)

### Getting Elements
Use `get(T, indices)` to read an element at a multi-dimensional coordinate.
The Zig scalar type must match the array dtype; a mismatch returns
`error.DTypeMismatch` instead of reinterpreting memory:

```zig
var mat = try num.zeros(allocator, .{ .shape = &.{ 3, 4 }, .dtype = .f64 });
defer mat.deinit();

// Set element at row 1, col 2 (works on contiguous arrays and strided views)
try mat.set(f64, &.{ 1, 2 }, 42.5);

// Read element back
const val = try mat.get(f64, &.{ 1, 2 }); // 42.5
```

### Direct Flat Indexing
For maximum throughput over contiguous memory, use typed slices:

```zig
var flat = try num.arange(allocator, .{ .start = 0, .stop = 4, .dtype = .f64 });
defer flat.deinit();

const slice = try flat.asSlice(f64);
slice[0] = 10.0;
const first = slice[0];
```

---

## 2. Multi-Dimensional Slicing (`slice`)

The `num.manip.slice` function creates a zero-allocation strided view from
per-dimension `Slice` descriptors (`start`/`stop` accept negative indices,
`step` may be negative for reversal):

```zig
const Slice = num.Slice;

// Given a 4x4 matrix
var m = try num.arange(allocator, .{ .start = 0, .stop = 16, .dtype = .f64 });
defer m.deinit();
var m4 = try num.manip.reshape(m, .{ .shape = &.{ 4, 4 } });
defer m4.deinit();

// Extract rows 1..3 and columns 1..3 (a 2x2 sub-matrix view)
var sub = try num.manip.slice(m4, &.{
    .{ .start = 1, .stop = 3 },
    .{ .start = 1, .stop = 3 },
});
defer sub.deinit();
```

---

## 3. Bounds Checking and Errors

Bounds checking in `num.zig` ensures safety against buffer overruns:

- Coordinates outside the valid dimensions return `error.IndexOutOfBounds`.
- A wrong number of indices returns `error.RankMismatch`.
- A Zig scalar type that does not match the array dtype returns `error.DTypeMismatch`.
- An invalid slice (e.g. zero step) returns `error.InvalidSlice` or `error.StepCannotBeZero`.
