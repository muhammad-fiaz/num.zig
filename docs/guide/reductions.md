# Reductions & Accumulations

Reductions collapse one or more axes of an array into scalar values or lower-dimensional projections.

---

## 1. Global Reductions

Global reductions aggregate all elements of an array into a single 0-dimensional scalar array:

```zig
var arr = try num.arange(allocator, .{ .start = 1.0, .stop = 11.0, .step = 1.0, .dtype = .f64 }); // 1..10
defer arr.deinit();

var s = try num.reduce.sum(arr, .{});   defer s.deinit(); // 55.0
var p = try num.reduce.prod(arr, .{});  defer p.deinit(); // 3628800.0
var m = try num.reduce.mean(arr, .{});  defer m.deinit(); // 5.5
var med = try num.reduce.median(arr, .{}); defer med.deinit(); // 5.5
var v = try num.reduce.variance(arr, .{}); defer v.deinit();
var sd = try num.reduce.stdDev(arr, .{}); defer sd.deinit();
var min_v = try num.reduce.min(arr, .{}); defer min_v.deinit(); // 1.0
var max_v = try num.reduce.max(arr, .{}); defer max_v.deinit(); // 10.0

const sum_val = try s.get(f64, &.{});
```

---

## 2. Axis-Wise Reductions

Reduce along a specific axis while maintaining or projecting dimensions:

```zig
var mat = try num.ones(allocator, .{ .shape = &.{ 3, 4 }, .dtype = .f64 });
defer mat.deinit();

// Sum along columns (axis 0) -> Shape [4], values = [3, 3, 3, 3]
var col_sums = try num.reduce.sum(mat, .{ .axis = 0 });
defer col_sums.deinit();

// Sum along rows (axis 1) -> Shape [3], values = [4, 4, 4]
var row_sums = try num.reduce.sum(mat, .{ .axis = 1 });
defer row_sums.deinit();
```

---

## 3. Extrema Indices (`argmin`, `argmax`)

Find the flat or multi-dimensional index of the minimum or maximum element:

```zig
var idx_min = try num.reduce.argmin(arr, .{});
defer idx_min.deinit();

var idx_max = try num.reduce.argmax(arr, .{});
defer idx_max.deinit();
```

---

## 4. Cumulative & Difference Operations

### `cumsum` & `cumprod`
Computes cumulative sums or products along an axis:
```zig
var cs = try num.reduce.cumsum(arr, .{ .axis = 0 });
defer cs.deinit();

var cp = try num.reduce.cumprod(arr, .{ .axis = 0 });
defer cp.deinit();
```

### `diff`
Computes $n$-th order discrete differences along an axis:
```zig
var d = try num.reduce.diff(arr, .{ .n = 1, .axis = 0 });
defer d.deinit();
```

---

## 5. Boolean Predicates

- **`all`**: Evaluates whether all elements evaluate to `true` / nonzero.
- **`any`**: Evaluates whether any element evaluates to `true` / nonzero.
- **`countNonzero`**: Counts the number of non-zero elements.
