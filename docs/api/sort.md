# Sorting & Searching API

Module: `@import("num").sort`

---

## Sorting

```zig
pub const SortOptions = struct {
    axis: ?isize = -1,   // null = global sort over all elements
    order: SortOrder = .asc,
    stable: bool = false,
};

/// Sort array in-place along the given axis.
pub fn sort(arr: *Array, options: SortOptions) !void;

/// Return a sorted copy of the array.
pub fn sorted(arr: Array, options: SortOptions) !Array;

pub const ArgsortOptions = struct {
    axis: isize = -1,
    order: SortOrder = .asc,
    stable: bool = false,
};

/// Return indices that would sort the array along the specified axis.
pub fn argsort(arr: Array, options: ArgsortOptions) !Array;
```

---

## Searching

```zig
/// Binary search for values `v` in a sorted array `a`. Returns insertion indices.
pub fn searchSorted(a: Array, v: Array) !Array;

/// Return flat indices of non-zero elements.
pub fn flatNonzero(allocator: std.mem.Allocator, a: Array) !Array;

/// Return per-axis indices of non-zero elements (like numpy.nonzero).
pub fn nonzero(allocator: std.mem.Allocator, a: Array) ![]Array;

/// Return (N, ndim) array of indices where condition is non-zero (like numpy.argwhere).
pub fn argwhere(allocator: std.mem.Allocator, a: Array) !Array;
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

pub fn unique(allocator: std.mem.Allocator, a: Array) !UniqueResult;
pub fn intersect1d(allocator: std.mem.Allocator, ar1: Array, ar2: Array) !Array;
pub fn union1d(allocator: std.mem.Allocator, ar1: Array, ar2: Array) !Array;
pub fn setdiff1d(allocator: std.mem.Allocator, ar1: Array, ar2: Array) !Array;
pub fn isin(allocator: std.mem.Allocator, element: Array, test_elements: Array) !Array;
```

