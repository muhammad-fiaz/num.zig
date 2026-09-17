# Shape Manipulation

`num.zig` offers a rich suite of functions for altering array shapes, axes, concatenating, stacking, padding, and tiling.

---

## 1. Reshape and Flatten

### `reshape` / `reshapeInPlace`
Changes the dimension of an array without changing its underlying data:

```zig
var a = try num.arange(allocator, f64, 0.0, 6.0, 1.0);
defer a.deinit();

// Reshape 1D [6] into 2D [2, 3]
try a.reshapeInPlace(&.{ 2, 3 });
```

### `flatten`
Collapses an N-dimensional array into a 1-D contiguous array:
```zig
var flat = try a.flatten();
defer flat.deinit();
```

---

## 2. Transposition and Axis Permutation

### `transpose`
Permutes axes. By default, reverses all axes:
```zig
// Inverts 2x3 matrix to 3x2
var t = try a.transpose();
defer t.deinit();
```

### `swapAxes` and `moveAxis`
Swaps two specific axes:
```zig
var s = try a.swapAxes(0, 1);
defer s.deinit();
```

---

## 3. Dimensional Squeeze and Expansion

### `squeeze`
Removes single-dimensional axes (dimensions equal to 1):
```zig
var tensor = try num.zeros(allocator, f64, &.{ 1, 3, 1, 5 });
defer tensor.deinit();

var sq = try tensor.squeeze(); // Shape becomes [3, 5]
defer sq.deinit();
```

### `expandDims`
Inserts a new axis of dimension 1 at a specified index:
```zig
var exp = try sq.expandDims(0); // Shape becomes [1, 3, 5]
defer exp.deinit();
```

---

## 4. Concatenation, Stacking, and Splitting

### `concat`
Joins multiple arrays along an existing axis:
```zig
var a1 = try num.ones(allocator, f64, &.{ 2, 3 });
defer a1.deinit();
var a2 = try num.zeros(allocator, f64, &.{ 2, 3 });
defer a2.deinit();

// Join along rows (axis 0) -> [4, 3]
var joined = try num.concat(allocator, f64, &.{ &a1, &a2 }, 0);
defer joined.deinit();
```

### `stack`
Joins a sequence of arrays along a new axis:
```zig
// Stack two [2, 3] arrays along new axis 0 -> [2, 2, 3]
var stacked = try num.stack(allocator, f64, &.{ &a1, &a2 }, 0);
defer stacked.deinit();
```

---

## 5. Tile, Repeat, Roll, and Pad

- **`tile`**: Replicates an array by the number of times given by reps.
- **`repeat`**: Repeats individual elements along an axis.
- **`roll`**: Rolls array elements along a given axis by a shift amount.
- **`pad`**: Pads an array along its boundaries with constant or edge values.
