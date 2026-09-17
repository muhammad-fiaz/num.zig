# Statistics API

Module: `@import("num").stats`

---

## Central Tendency & Dispersion

```zig
pub fn mean(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn min(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn max(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn range(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
pub fn median(arr: Array, options: struct { axis: ?isize = null, keepDims: bool = false }) !Array;
pub fn variance(arr: Array, options: struct { axis: ?isize = null, ddof: usize = 0, keepDims: bool = false, dtype: ?DType = null }) !Array;
pub fn stdDev(arr: Array, options: struct { axis: ?isize = null, ddof: usize = 0, keepDims: bool = false, dtype: ?DType = null }) !Array;
```

`mean` shares semantics with `num.reduce.mean`. `median`, `variance`, and `stdDev` share result semantics with their `num.reduce` counterparts; the `stats` variants additionally support `ddof` (variance/stdDev) for degrees-of-freedom correction.

---

## Quantiles & Order Statistics

```zig
pub fn quantile(arr: Array, q: f64) !Array;
pub fn quantileWithOptions(arr: Array, q: f64, options: struct { axis: ?isize = null, keepDims: bool = false, method: QuantileMethod = .linear }) !Array;
pub fn percentile(arr: Array, q: f64, options: struct { axis: ?isize = null, keepDims: bool = false, method: QuantileMethod = .linear }) !Array;
pub const QuantileMethod = enum { linear, lower, higher, midpoint, nearest };
```

---

## Covariance and Correlation

```zig
pub fn covariance(x: Array, y: ?Array, options: struct { rowvar: bool = true, bias: bool = false, ddof: ?usize = null }) !Array;
pub fn corrcoef(x: Array, y: ?Array, options: struct { rowvar: bool = true, bias: bool = false, ddof: ?usize = null }) !Array;
```

---

## Histograms

```zig
pub const HistogramResult = struct {
    counts: Array,
    bin_edges: Array,
    pub fn deinit(self: *HistogramResult) void;
};

pub fn histogram(
    arr: Array,
    options: struct {
        bins: usize = 10,
        range: ?[2]f64 = null,
        density: bool = false,
    },
) !HistogramResult;
```

