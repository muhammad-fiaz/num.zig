# Indexing & Slicing

`num.zig` provides flexible indexing primitives for multi-dimensional scalar access, sub-tensor slicing, and strided views.

---

## 1. Scalar Indexing (`get` and `set`)

### Getting Elements
Use `get(T, indices)` to read an element at a multi-dimensional coordinate:

```zig
var mat = try num.zeros(allocator, f64, &.{ 3, 4 });
defer mat.deinit();

// Set element at row 1, col 2
try mat.set(f64, &.{ 1, 2 }, 42.5);

// Read element back
const val = try mat.get(f64, &.{ 1, 2 }); // 42.5
```

### Direct Flat Indexing
For maximum throughput when iterating over contiguous memory, use flat indexing:

```zig
mat.data(f64)[0] = 10.0;
const first = mat.data(f64)[0];
```

---

## 2. Multi-Dimensional Slicing (`slice`)

The `slice` method creates a new sub-array extracted from specified bounds along each dimension:

```zig
const SliceSpec = struct {
    start: usize = 0,
    end: ?usize = null,
    step: usize = 1,
};
```

### Example: Sub-matrix Extraction
```zig
// Given a 4x4 matrix
var m = try num.arange(allocator, f64, 0.0, 16.0, 1.0);
defer m.deinit();
try m.reshapeInPlace(&.{ 4, 4 });

// Extract rows 1..3 and columns 1..3 (a 2x2 sub-matrix)
var sub = try m.slice(&.{
    .{ .start = 1, .end = 3 },
    .{ .start = 1, .end = 3 },
});
defer sub.deinit();
```

---

## 3. Negative Indices and Strides

Bounds checking in `num.zig` ensures safety against buffer overruns. Coordinates outside the valid dimensions return `error.IndexOutOfBounds`.
