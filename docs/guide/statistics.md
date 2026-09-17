# Statistics & Correlation

`num.zig` provides statistical distributions, order statistics, moments, variance, covariance, and correlation metrics.

---

## 1. Central Tendency & Dispersion

```zig
var data = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0 },
    .shape = &.{10},
});
defer data.deinit();

// Arithmetic Mean
var m = try num.stats.mean(data, .{});
defer m.deinit();

// Median (resilient to outliers)
var med = try num.stats.median(data, .{});
defer med.deinit();

// Sample and Population Variance (ddof = 1 or ddof = 0)
var v = try num.stats.variance(data, .{ .ddof = 1 });
defer v.deinit();

// Standard Deviation
var sd = try num.stats.stdDev(data, .{ .ddof = 1 });
defer sd.deinit();

// Minimum, maximum, and peak-to-peak range
var lo = try num.stats.min(data, .{});
defer lo.deinit();
var hi = try num.stats.max(data, .{});
defer hi.deinit();
var r = try num.stats.range(data, .{});
defer r.deinit();
```

---

## 2. Percentiles and Quantiles

Compute arbitrary percentiles using linear interpolation (default), with `lower`, `higher`, `midpoint`, and `nearest` methods available via `.method`:

```zig
// 25th, 50th, and 75th percentiles (quartiles)
var q25 = try num.stats.percentile(data, 25.0, .{}); defer q25.deinit();
var q50 = try num.stats.percentile(data, 50.0, .{}); defer q50.deinit();
var q75 = try num.stats.percentile(data, 75.0, .{}); defer q75.deinit();
```

---

## 3. Covariance & Pearson Correlation

### `covariance`
Computes the covariance matrix between multiple variables:
```zig
var cov_mat = try num.stats.covariance(data_matrix, null, .{});
defer cov_mat.deinit();
```

### `corrcoef`
Computes the Pearson product-moment correlation coefficients matrix:
```zig
var corr_mat = try num.stats.corrcoef(data_matrix, null, .{});
defer corr_mat.deinit();
```

---

## 4. Histograms and Binning

Partition 1D data into uniform or arbitrary histogram bins:

```zig
// Partition data into 10 uniform bins
var hist = try num.stats.histogram(data, .{ .bins = 10 });
defer hist.deinit();

std.debug.print("Counts: {d}, Bin edges: {d}\n", .{
    hist.counts.elementCount(),
    hist.bin_edges.elementCount(),
});
```
