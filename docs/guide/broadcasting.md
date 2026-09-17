# Broadcasting

Broadcasting allows arithmetic operations between arrays of different shapes without copying data when their dimensions are mutually compatible.

---

## 1. Broadcasting Semantics

Two dimensions are compatible if:
1. They are equal, or
2. One of them is `1`.

When comparing shapes, dimensions are aligned from right to left (trailing axes first). Missing leading dimensions are prepended with `1`.

### Examples
- `[3, 4]` and `[4]` -> `[3, 4]` (Compatible)
- `[2, 1, 5]` and `[3, 5]` -> `[2, 3, 5]` (Compatible)
- `[2, 3]` and `[2, 2]` -> **Incompatible** (`error.IncompatibleShapes`)

---

## 2. Zero-Copy Broadcasting (`broadcastTo`)

You can create a broadcasted view of an array where dimensions with size 1 are expanded to larger sizes with **zero memory copies** by setting their stride to 0:

```zig
var v = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 1.0, 2.0, 3.0 },
    .shape = &.{ 1, 3 },
});
defer v.deinit();

// Broadcast [1, 3] to [4, 3] without allocating backing storage
var view = try v.broadcastTo(&.{ 4, 3 });
defer view.deinit();
```

---

## 3. Automatic Broadcasting in Binary Ops

All binary arithmetic operations (`add`, `subtract`, `multiply`, `divide`, etc.) automatically perform broadcasting between the left and right operands:

```zig
// Matrix [2, 3]
var a = try num.ones(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
defer a.deinit();

// Bias vector [3]
var b = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 10.0, 20.0, 30.0 },
    .shape = &.{3},
});
defer b.deinit();

// Result is [2, 3], broadcasting b across both rows
var c = try num.ops.add(a, b, .{});
defer c.deinit();
```
