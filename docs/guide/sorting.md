# Sorting & Set Operations

`num.zig` provides fast intro-sort routines, indirect argument sorting, binary search utilities, and mathematical 1D set operations.

---

## 1. Array Sorting

### `sort` (In-Place)
Sorts array elements in-place along a specified axis:
```zig
var arr = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 9.0, 1.0, 4.0, 3.0, 7.0 },
    .shape = &.{5},
});
defer arr.deinit();

try num.sort.sort(&arr, .{ .axis = 0 }); // arr is now [1.0, 3.0, 4.0, 7.0, 9.0]
```

### `sorted` (Copy)
Returns a new sorted copy without mutating the original array:
```zig
var s = try num.sort.sorted(arr, .{ .axis = 0 });
defer s.deinit();
```

---

## 2. Argument Sorting (`argsort`)

Returns an array of indices that would sort the array along a given axis:

```zig
var indices = try num.sort.argsort(arr, .{ .axis = 0 });
defer indices.deinit();
```

---

## 3. Binary Search (`searchSorted`)

Finds the insertion indices where elements should be inserted into a sorted array to maintain order:

```zig
var sorted_arr = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 10.0, 20.0, 30.0, 40.0 },
    .shape = &.{4},
});
defer sorted_arr.deinit();

var queries = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 25.0, 5.0, 45.0 },
    .shape = &.{3},
});
defer queries.deinit();

var ins_idx = try num.sort.searchSorted(sorted_arr, queries, .{ .side = .left });
defer ins_idx.deinit(); // [2, 0, 4]
```

---

## 4. Set Operations

Perform mathematical set theory operations on 1D arrays:

### `unique`
Extracts unique sorted elements (optionally returning occurrence counts and inverse indices):
```zig
var u = try num.sort.unique(arr, .{});
defer u.deinit();
```

### Set Intersections and Differences
- **`intersect1d`**: Common elements between two arrays.
- **`union1d`**: Unique union of elements from both arrays.
- **`setdiff1d`**: Set difference (elements in A not in B).
- **`isin`**: Tests whether each element in A is present in B.
