const std = @import("std");
const core = @import("../core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

pub const SortAlgo = enum {
    QuickSort,
    MergeSort,
    HeapSort,
    InsertionSort,
    BubbleSort,
    TimSort, // Default std.sort
};

/// Sorts an array using a specific algorithm.
pub fn sortByAlgo(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize, algo: SortAlgo) !NDArray(T) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;
    var result = try arr.copy();
    errdefer result.deinit();

    // Iterate over all dimensions except axis
    var iter_shape = try allocator.alloc(usize, arr.rank());
    defer allocator.free(iter_shape);
    @memcpy(iter_shape, arr.shape);
    iter_shape[axis] = 1;

    var iter = try core.NdIterator.init(allocator, iter_shape);
    defer iter.deinit();

    const dim_size = arr.shape[axis];
    var slice_vals = try allocator.alloc(T, dim_size);
    defer allocator.free(slice_vals);

    var coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(coords);

    while (iter.next()) |iter_coords| {
        @memcpy(coords, iter_coords);

        // Read slice
        for (0..dim_size) |i| {
            coords[axis] = i;
            slice_vals[i] = try result.get(coords);
        }

        // Sort slice based on algo
        switch (algo) {
            .QuickSort => quickSort(T, slice_vals, 0, @intCast(slice_vals.len - 1)),
            .MergeSort => try mergeSort(allocator, T, slice_vals),
            .HeapSort => heapSort(T, slice_vals),
            .InsertionSort => insertionSort(T, slice_vals),
            .BubbleSort => bubbleSort(T, slice_vals),
            .TimSort => std.mem.sort(T, slice_vals, {}, std.sort.asc(T)),
        }

        // Write back
        for (0..dim_size) |i| {
            coords[axis] = i;
            try result.set(coords, slice_vals[i]);
        }
    }
    return result;
}

fn quickSort(comptime T: type, arr: []T, low: isize, high: isize) void {
    if (low < high) {
        const pi = partition_qs(T, arr, low, high);
        quickSort(T, arr, low, pi - 1);
        quickSort(T, arr, pi + 1, high);
    }
}

fn partition_qs(comptime T: type, arr: []T, low: isize, high: isize) isize {
    const pivot = arr[@intCast(high)];
    var i = low - 1;
    var j = low;
    while (j < high) : (j += 1) {
        if (arr[@intCast(j)] < pivot) {
            i += 1;
            std.mem.swap(T, &arr[@intCast(i)], &arr[@intCast(j)]);
        }
    }
    std.mem.swap(T, &arr[@intCast(i + 1)], &arr[@intCast(high)]);
    return i + 1;
}

fn mergeSort(allocator: Allocator, comptime T: type, arr: []T) !void {
    if (arr.len <= 1) return;
    const mid = arr.len / 2;
    const left = try allocator.dupe(T, arr[0..mid]);
    defer allocator.free(left);
    const right = try allocator.dupe(T, arr[mid..]);
    defer allocator.free(right);

    try mergeSort(allocator, T, left);
    try mergeSort(allocator, T, right);

    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;

    while (i < left.len and j < right.len) {
        if (left[i] <= right[j]) {
            arr[k] = left[i];
            i += 1;
        } else {
            arr[k] = right[j];
            j += 1;
        }
        k += 1;
    }

    while (i < left.len) {
        arr[k] = left[i];
        i += 1;
        k += 1;
    }
    while (j < right.len) {
        arr[k] = right[j];
        j += 1;
        k += 1;
    }
}

fn heapSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    var i = @as(isize, @intCast(n / 2 - 1));
    while (i >= 0) : (i -= 1) {
        heapify(T, arr, n, @intCast(i));
    }
    var j = @as(isize, @intCast(n - 1));
    while (j > 0) : (j -= 1) {
        std.mem.swap(T, &arr[0], &arr[@intCast(j)]);
        heapify(T, arr, @intCast(j), 0);
    }
}

fn heapify(comptime T: type, arr: []T, n: usize, i: usize) void {
    var largest = i;
    const l = 2 * i + 1;
    const r = 2 * i + 2;

    if (l < n and arr[l] > arr[largest]) largest = l;
    if (r < n and arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        std.mem.swap(T, &arr[i], &arr[largest]);
        heapify(T, arr, n, largest);
    }
}

fn insertionSort(comptime T: type, arr: []T) void {
    for (1..arr.len) |i| {
        const key = arr[i];
        var j: isize = @intCast(i - 1);
        while (j >= 0 and arr[@intCast(j)] > key) : (j -= 1) {
            arr[@intCast(j + 1)] = arr[@intCast(j)];
        }
        arr[@intCast(j + 1)] = key;
    }
}

fn bubbleSort(comptime T: type, arr: []T) void {
    const n = arr.len;
    for (0..n) |i| {
        var swapped = false;
        for (0..n - i - 1) |j| {
            if (arr[j] > arr[j + 1]) {
                std.mem.swap(T, &arr[j], &arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

/// Sorts an array along a given axis.
///
/// Returns a new sorted array (copy). The original array is not modified.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input NDArray.
///     axis: The axis along which to sort.
///
/// Returns:
///     A new NDArray containing the sorted data.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{3.0, 1.0, 2.0});
/// defer a.deinit();
///
/// var sorted = try sort.sort(allocator, f32, a, 0);
/// defer sorted.deinit();
/// // sorted is {1.0, 2.0, 3.0}
/// ```
pub fn sort(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(T) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;

    var result = try arr.copy(allocator);

    // Iterate over all dimensions except axis
    // For each 1D slice along axis, sort it.

    // We can use NdIterator on a shape that has 1 at axis.
    var iter_shape = try allocator.alloc(usize, arr.rank());
    defer allocator.free(iter_shape);
    @memcpy(iter_shape, arr.shape);
    iter_shape[axis] = 1;

    var iter = try core.NdIterator.init(allocator, iter_shape);
    defer iter.deinit();

    const dim_size = arr.shape[axis];
    var slice_vals = try allocator.alloc(T, dim_size);
    defer allocator.free(slice_vals);

    var coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(coords);

    while (iter.next()) |iter_coords| {
        @memcpy(coords, iter_coords);

        // Read slice
        for (0..dim_size) |i| {
            coords[axis] = i;
            slice_vals[i] = try result.get(coords);
        }

        // Sort slice
        std.mem.sort(T, slice_vals, {}, std.sort.asc(T));

        // Write back
        for (0..dim_size) |i| {
            coords[axis] = i;
            try result.set(coords, slice_vals[i]);
        }
    }

    return result;
}

/// Sorts a 1D array.
pub fn sort1D(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(T) {
    _ = allocator;
    if (arr.rank() != 1) return core.Error.RankMismatch;
    const result = try arr.copy();
    std.sort.block(T, result.data, {}, std.sort.asc(T));
    return result;
}

/// Returns the indices that would sort an array.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input NDArray.
///     axis: The axis along which to sort.
///
/// Returns:
///     A new NDArray(usize) containing the indices.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{3.0, 1.0, 2.0});
/// defer a.deinit();
///
/// var indices = try sort.argsort(allocator, f32, a, 0);
/// defer indices.deinit();
/// // indices is {1, 2, 0}
/// ```
pub fn argsort(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(usize) {
    if (axis >= arr.rank()) return core.Error.IndexOutOfBounds;

    var result = try NDArray(usize).init(allocator, arr.shape);

    var iter_shape = try allocator.alloc(usize, arr.rank());
    defer allocator.free(iter_shape);
    @memcpy(iter_shape, arr.shape);
    iter_shape[axis] = 1;

    var iter = try core.NdIterator.init(allocator, iter_shape);
    defer iter.deinit();

    const dim_size = arr.shape[axis];

    // Struct to hold value and original index
    const Pair = struct { val: T, idx: usize };

    var slice_vals = try allocator.alloc(Pair, dim_size);
    defer allocator.free(slice_vals);

    var coords = try allocator.alloc(usize, arr.rank());
    defer allocator.free(coords);

    while (iter.next()) |iter_coords| {
        @memcpy(coords, iter_coords);

        // Read slice
        for (0..dim_size) |i| {
            coords[axis] = i;
            slice_vals[i] = .{ .val = try arr.get(coords), .idx = i };
        }

        // Sort slice
        const sortFn = struct {
            fn lessThan(_: void, lhs: Pair, rhs: Pair) bool {
                return lhs.val < rhs.val;
            }
        }.lessThan;

        std.mem.sort(Pair, slice_vals, {}, sortFn);

        // Write indices back
        for (0..dim_size) |i| {
            coords[axis] = i;
            try result.set(coords, slice_vals[i].idx);
        }
    }

    return result;
}

/// Returns the indices that would sort a 1D array.
pub fn argsort1D(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize) {
    if (arr.rank() != 1) return core.Error.RankMismatch;

    const indices = try allocator.alloc(usize, arr.size());
    defer allocator.free(indices);

    for (0..arr.size()) |i| indices[i] = i;

    const Context = struct {
        data: []const T,
        pub fn less(ctx: @This(), a: usize, b: usize) bool {
            return ctx.data[a] < ctx.data[b];
        }
    };

    std.sort.block(usize, indices, Context{ .data = arr.data }, Context.less);

    const result = try NDArray(usize).init(allocator, arr.shape);
    @memcpy(result.data, indices);
    return result;
}

/// Return the indices of the elements that are non-zero.
/// Returns an array of shape (ndim, count) containing indices for each dimension.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The data type of the array elements.
///     arr: The input NDArray.
///
/// Returns:
///     A new NDArray(usize) containing the indices of non-zero elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{1.0, 0.0, 2.0});
/// defer a.deinit();
///
/// var nz = try sort.nonzero(allocator, f32, a);
/// defer nz.deinit();
/// // nz shape is (1, 2), data is {0, 2}
/// ```
pub fn nonzero(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize) {
    // Count non-zeros
    var count: usize = 0;
    var iter = try core.NdIterator.init(allocator, arr.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        const val = try arr.get(coords);
        if (val != 0) count += 1;
    }

    var result = try NDArray(usize).init(allocator, &.{ arr.rank(), count });

    iter.reset();
    var idx: usize = 0;
    while (iter.next()) |coords| {
        const val = try arr.get(coords);
        if (val != 0) {
            for (coords, 0..) |c, d| {
                // Store as (ndim, count) -> row d, col idx
                // stride[0] = count, stride[1] = 1
                result.data[d * count + idx] = c;
            }
            idx += 1;
        }
    }
    return result;
}

/// Return indices that are non-zero in the flattened version of a.
pub fn flatnonzero(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize) {
    var count: usize = 0;
    var iter = try core.NdIterator.init(allocator, arr.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        const val = try arr.get(coords);
        if (val != 0) count += 1;
    }

    var result = try NDArray(usize).init(allocator, &.{count});

    iter.reset();
    var idx: usize = 0;
    var flat_idx: usize = 0;
    while (iter.next()) |coords| {
        const val = try arr.get(coords);
        if (val != 0) {
            result.data[idx] = flat_idx;
            idx += 1;
        }
        flat_idx += 1;
    }
    return result;
}

test "sort argsort 1d" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{4});
    defer a.deinit();
    try a.set(&.{0}, 3);
    try a.set(&.{1}, 1);
    try a.set(&.{2}, 4);
    try a.set(&.{3}, 2);

    var sorted = try sort1D(allocator, f32, a);
    defer sorted.deinit();
    try std.testing.expectEqual(try sorted.get(&.{0}), 1.0);
    try std.testing.expectEqual(try sorted.get(&.{3}), 4.0);

    var indices = try argsort1D(allocator, f32, a);
    defer indices.deinit();
    // Should be [1, 3, 0, 2] -> values [1, 2, 3, 4]
    try std.testing.expectEqual(try indices.get(&.{0}), 1);
    try std.testing.expectEqual(try indices.get(&.{1}), 3);
}

/// Return a partitioned copy of an array.
///
/// Creates a copy of the array with its elements rearranged in such a way that
/// the value of the element in k-th position is in the position it would be in
/// a sorted array. All elements smaller than the k-th element are moved
/// before this element and all equal or greater are moved behind it.
/// The ordering of the elements in the two partitions is undefined.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     arr: Input array.
///     kth: Element index to partition by.
///     axis: Axis along which to sort.
///
/// Returns:
///     Partitioned array.
pub fn partition(allocator: Allocator, comptime T: type, arr: NDArray(T), kth: usize, axis: usize) !NDArray(T) {
    _ = kth; // Unused for now as we do full sort
    // For now, we just sort, which satisfies the partition requirement.
    return sort(allocator, T, arr, axis);
}

/// Perform an indirect partition along the given axis.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     arr: Input array.
///     kth: Element index to partition by.
///     axis: Axis along which to sort.
///
/// Returns:
///     Array of indices that partition the data.
pub fn argpartition(allocator: Allocator, comptime T: type, arr: NDArray(T), kth: usize, axis: usize) !NDArray(usize) {
    _ = kth; // Unused for now as we do full sort
    // For now, we just argsort.
    return argsort(allocator, T, arr, axis);
}
