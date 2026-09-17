# Shape Manipulation

`num.zig` offers a rich suite of functions for altering array shapes, axes, concatenating, stacking, padding, and tiling. View-returning operations (`reshape`, `transpose`, `swapAxes`, `moveAxis`, `squeeze`, `expandDims`, `slice`) are zero-allocation; copy-producing operations (`flatten`, `concat`, `stack`, `tile`, `repeat`, `pad`, `append`, `insert`, `delete`) return new owned arrays.

---

## 1. Reshape and Flatten

### `reshape`
Changes the dimensions of an array without changing its underlying data (view when contiguous):

```zig
var a = try num.arange(allocator, .{ .start = 0.0, .stop = 6.0, .step = 1.0, .dtype = .f64 });
defer a.deinit();

// Reshape 1D [6] into 2D [2, 3]
var m = try num.manip.reshape(a, .{ .shape = &.{ 2, 3 } });
defer m.deinit();
```

### `flatten` / `ravel`
```zig
// Contiguous 1D copy, always allocating
var flat = try num.manip.flatten(a);
defer flat.deinit();

// 1D view when contiguous, copy otherwise
var r = try num.manip.ravel(a);
defer r.deinit();
```

---

## 2. Transposition and Axis Permutation

### `transpose`
Permutes axes. By default, reverses all axes (view):
```zig
// Transpose 2x3 matrix to 3x2
var t = try num.manip.transpose(m, .{});
defer t.deinit();
```

### `swapAxes` and `moveAxis`
```zig
var s = try num.manip.swapAxes(m, 0, 1);
defer s.deinit();

var mv = try num.manip.moveAxis(m, 0, 1);
defer mv.deinit();
```

---

## 3. Dimensional Squeeze and Expansion

### `squeeze`
Removes single-dimensional axes (dimensions equal to 1):
```zig
var tensor = try num.zeros(allocator, .{ .shape = &.{ 1, 3, 1, 5 }, .dtype = .f64 });
defer tensor.deinit();

var sq = try num.manip.squeeze(tensor, .{}); // Shape becomes [3, 5]
defer sq.deinit();
```

### `expandDims`
Inserts a new axis of dimension 1 at a specified index:
```zig
var exp = try num.manip.expandDims(sq, .{ .axis = 0 }); // Shape becomes [1, 3, 5]
defer exp.deinit();
```

---

## 4. Concatenation, Stacking, and Splitting

### `concat`
Joins multiple arrays along an existing axis:
```zig
var a1 = try num.ones(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
defer a1.deinit();
var a2 = try num.zeros(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
defer a2.deinit();

// Join along rows (axis 0) -> [4, 3]
const pair = [_]num.Array{ a1, a2 };
var joined = try num.manip.concat(&pair, .{ .axis = 0 });
defer joined.deinit();
```

### `stack`
Joins a sequence of arrays along a new axis:
```zig
// Stack two [2, 3] arrays along new axis 0 -> [2, 2, 3]
var stacked = try num.manip.stack(&pair, .{ .axis = 0 });
defer stacked.deinit();
```

### `append`, `insert`, `delete`
```zig
// Append values (flattened when axis is null)
var ap = try num.manip.append(a1, a2, .{});
defer ap.deinit();

// Insert a2 into a1 at index 1
var ins = try num.manip.insert(a1, 1, a2, .{ .axis = 0 });
defer ins.deinit();

// Delete index 0 along axis 0
var del = try num.manip.delete(joined, 0, .{ .axis = 0 });
defer del.deinit();
```

---

## 5. Tile, Repeat, Roll, and Pad

- **`tile`**: Replicates an array by the number of times given by reps: `try num.manip.tile(a, .{ .reps = &.{3} })`.
- **`repeat`**: Repeats individual elements along an axis: `try num.manip.repeat(a, .{ .repeats = 2 })`.
- **`roll`**: Rolls array elements along a given axis by a shift amount: `try num.manip.roll(a, .{ .shift = 1, .axis = 0 })`.
- **`pad`**: Pads an array with a constant value: `try num.manip.pad(a, .{ .pad_width = &.{ .{ 1, 1 } } })`.
