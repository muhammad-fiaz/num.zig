# Reductions & Accumulations API

Module: `@import("num").reduce`

---

## Global & Axis-Wise Reductions

All reductions accept an optional inline configuration structure:

```zig
.{
    .axis: ?isize = null,       // target axis or null for global reduction
    .keepDims: bool = false,    // retain singleton dimensions
    .dtype: ?DType = null,      // accumulator / return data type
}
```

```zig
pub fn sum(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn prod(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn min(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn max(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn mean(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn all(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
pub fn any(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
```

---

## Index Reductions

```zig
pub fn argmin(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
pub fn argmax(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
pub fn countNonzero(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
```

---

## Accumulations & Differences

```zig
pub fn cumsum(arr: Array, options: struct { axis: ?isize = null, dtype: ?DType = null }) !Array;
pub fn cumprod(arr: Array, options: struct { axis: ?isize = null, dtype: ?DType = null }) !Array;
pub fn cummin(arr: Array, options: struct { axis: ?isize = null }) !Array;
pub fn cummax(arr: Array, options: struct { axis: ?isize = null }) !Array;
pub fn diff(arr: Array, options: struct { n: usize = 1, axis: ?isize = null }) !Array;
```

