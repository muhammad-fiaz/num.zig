# Sorting & Searching API

Module: `@import("num").sort`

---

## Sorting

```zig
/// Sort array in-place along the given axis.
/// Inline config: .{ .axis = -1, .order = .asc, .stable = false } (null axis = global sort).
pub fn sort(arr: *Array, options: struct { axis: ?isize = -1, order: SortOrder = .asc, stable: bool = false }) !void;

/// Return a sorted copy of the array.
pub fn sorted(arr: Array, options: struct { axis: ?isize = -1, order: SortOrder = .asc, stable: bool = false }) !Array;

/// Return indices that would sort the array along the specified axis.
pub fn argsort(arr: Array, options: struct { axis: isize = -1, order: SortOrder = .asc, stable: bool = false }) !Array;
```

---

## Searching

```zig
/// Binary search for values `v` in a sorted array `a`. Returns insertion indices.
pub fn searchSorted(a: Array, v: Array) !Array;

/// Return flat indices of non-zero elements.
pub fn flatNonzero(a: Array) !Array;

/// Return per-axis indices of non-zero elements.
pub fn nonzero(a: Array) !Array;

/// Return (N, ndim) array of indices where condition is non-zero.
pub fn argwhere(a: Array) !Array;
```

---

## Set Operations

```zig
pub const UniqueResult = struct {
    values: Array,
    indices: ?Array = null,
    inverse: ?Array = null,
    counts: ?Array = null,
    pub fn deinit(self: *UniqueResult) void;
};

pub fn unique(arr: Array, options: struct { returnIndex: bool = false, returnInverse: bool = false, returnCounts: bool = false }) !UniqueResult;
pub fn intersect1d(ar1: Array, ar2: Array) !Array;
pub fn union1d(ar1: Array, ar2: Array) !Array;
pub fn setdiff1d(ar1: Array, ar2: Array) !Array;
pub fn isin(element: Array, test_elements: Array) !Array;
```

