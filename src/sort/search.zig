//! Search, unique element extraction, and non-zero index discovery.
//!
//! Provides binary search sorted insertion, unique values with counts/inverses,
//! and non-zero coordinate extraction.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

const ravel = @import("../manip/reshape.zig").ravel;

pub const SearchSide = enum {
    left,
    right,
};

const SearchSortedOptions = struct {
    side: SearchSide = .left,
};

/// Finds indices where elements should be inserted to maintain order.
pub fn searchSorted(
    sorted_arr: Array,
    values: Array,
    options: SearchSortedOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (sorted_arr.ndim != 1) return ShapeError.InvalidDimension;
    const n = sorted_arr.shape_dims[0];

    const sorted_buf = try sorted_arr.allocator.alloc(f64, n);
    defer sorted_arr.allocator.free(sorted_buf);

    for (0..n) |i| {
        sorted_buf[i] = try sorted_arr.getAsFloat(&.{i});
    }

    var out = try empty(values.allocator, .{
        .shape = values.shapeSlice(),
        .dtype = .i64,
    });
    errdefer out.deinit();

    const m = values.elementCount();
    var flat_vals = try ravel(values);
    defer flat_vals.deinit();

    var flat_out = try ravel(out);
    defer flat_out.deinit();

    const OrderCtx = struct {
        fn orderFn(context: f64, item: f64) std.math.Order {
            return std.math.order(context, item);
        }
    };

    for (0..m) |i| {
        const v = try flat_vals.getAsFloat(&.{i});
        const idx = if (options.side == .left)
            std.sort.lowerBound(f64, sorted_buf, v, OrderCtx.orderFn)
        else
            std.sort.upperBound(f64, sorted_buf, v, OrderCtx.orderFn);

        try flat_out.set(i64, &.{i}, @intCast(idx));
    }

    return out;
}

const UniqueOptions = struct {
    returnIndex: bool = false,
    returnInverse: bool = false,
    returnCounts: bool = false,
};

pub const UniqueResult = struct {
    values: Array,
    indices: ?Array = null,
    inverse: ?Array = null,
    counts: ?Array = null,

    pub fn deinit(self: *UniqueResult) void {
        self.values.deinit();
        if (self.indices) |*idx| idx.deinit();
        if (self.inverse) |*inv| inv.deinit();
        if (self.counts) |*cnt| cnt.deinit();
    }
};

/// Finds the sorted unique elements of an array.
pub fn unique(
    arr: Array,
    options: UniqueOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!UniqueResult {
    var flat = try ravel(arr);
    defer flat.deinit();

    const N = flat.elementCount();
    if (N == 0) {
        const empty_vals = try empty(arr.allocator, .{ .shape = &.{0}, .dtype = arr.dtype });
        return UniqueResult{ .values = empty_vals };
    }

    const items = try arr.allocator.alloc(f64, N);
    defer arr.allocator.free(items);

    const orig_indices = try arr.allocator.alloc(usize, N);
    defer arr.allocator.free(orig_indices);

    for (0..N) |i| {
        items[i] = try flat.getAsFloat(&.{i});
        orig_indices[i] = i;
    }

    // Sort original indices by items
    const Context = struct {
        vals: []const f64,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.vals[a] < ctx.vals[b];
        }
    };
    std.sort.pdq(usize, orig_indices, Context{ .vals = items }, Context.lessThan);

    // Count unique elements
    var n_unique: usize = 1;
    for (1..N) |i| {
        const prev_val = items[orig_indices[i - 1]];
        const cur_val = items[orig_indices[i]];
        if (prev_val != cur_val) {
            n_unique += 1;
        }
    }

    var unique_vals = try empty(arr.allocator, .{ .shape = &.{n_unique}, .dtype = arr.dtype });
    errdefer unique_vals.deinit();

    var unique_indices: ?Array = if (options.returnIndex)
        try empty(arr.allocator, .{ .shape = &.{n_unique}, .dtype = .i64 })
    else
        null;
    errdefer if (unique_indices) |*idx| idx.deinit();

    var unique_counts: ?Array = if (options.returnCounts)
        try zeros(arr.allocator, .{ .shape = &.{n_unique}, .dtype = .i64 })
    else
        null;
    errdefer if (unique_counts) |*cnt| cnt.deinit();

    var unique_inverse: ?Array = if (options.returnInverse)
        try empty(arr.allocator, .{ .shape = &.{N}, .dtype = .i64 })
    else
        null;
    errdefer if (unique_inverse) |*inv| inv.deinit();

    var u_idx: usize = 0;
    var cur_count: i64 = 0;

    for (0..N) |i| {
        const sorted_orig_idx = orig_indices[i];
        const val = items[sorted_orig_idx];

        if (i > 0 and val != items[orig_indices[i - 1]]) {
            if (unique_counts) |*cnt| {
                try cnt.set(i64, &.{u_idx}, cur_count);
            }
            u_idx += 1;
            cur_count = 0;
        }

        if (cur_count == 0) {
            try unique_vals.setFromFloat(&.{u_idx}, val);
            if (unique_indices) |*idx| {
                try idx.set(i64, &.{u_idx}, @intCast(sorted_orig_idx));
            }
        }

        if (unique_inverse) |*inv| {
            try inv.set(i64, &.{sorted_orig_idx}, @intCast(u_idx));
        }

        cur_count += 1;
    }

    if (unique_counts) |*cnt| {
        try cnt.set(i64, &.{u_idx}, cur_count);
    }

    return UniqueResult{
        .values = unique_vals,
        .indices = unique_indices,
        .inverse = unique_inverse,
        .counts = unique_counts,
    };
}

/// Returns the flat/linear indices of elements that are non-zero.
pub fn flatNonzero(
    arr: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var flat = try ravel(arr);
    defer flat.deinit();

    const N = flat.elementCount();
    var count: usize = 0;

    for (0..N) |i| {
        const val = try flat.getAsFloat(&.{i});
        if (val != 0.0) count += 1;
    }

    var out = try empty(arr.allocator, .{ .shape = &.{count}, .dtype = .i64 });
    errdefer out.deinit();

    var write_idx: usize = 0;
    for (0..N) |i| {
        const val = try flat.getAsFloat(&.{i});
        if (val != 0.0) {
            try out.set(i64, &.{write_idx}, @intCast(i));
            write_idx += 1;
        }
    }

    return out;
}

/// Returns multidimensional coordinates of elements that are non-zero.
/// Returns a 2D array of shape [count, ndim].
pub fn nonzero(
    arr: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return argwhere(arr);
}

/// Find the indices of array elements that are non-zero, grouped by element.
/// Returns a 2D array of shape [count, ndim].
pub fn argwhere(
    arr: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = arr.shape();
    var it = @import("../core/iterator.zig").NdIterator.init(s, arr.strides());
    var count: usize = 0;
    while (it.next()) |item| {
        const val = try arr.getAsFloat(item.indices[0..s.ndim]);
        if (val != 0.0) count += 1;
    }

    var out = try empty(arr.allocator, .{ .shape = &.{ count, s.ndim }, .dtype = .i64 });
    errdefer out.deinit();

    var it2 = @import("../core/iterator.zig").NdIterator.init(s, arr.strides());
    var row: usize = 0;
    while (it2.next()) |item| {
        const val = try arr.getAsFloat(item.indices[0..s.ndim]);
        if (val != 0.0) {
            for (0..s.ndim) |dim| {
                try out.set(i64, &.{ row, dim }, @intCast(item.indices[dim]));
            }
            row += 1;
        }
    }

    return out;
}

/// Find the sorted, unique intersection of two 1D arrays.
pub fn intersect1d(
    a: Array,
    b: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var u_a = try unique(a, .{});
    defer u_a.deinit();
    var u_b = try unique(b, .{});
    defer u_b.deinit();

    const n_a = u_a.values.elementCount();
    const n_b = u_b.values.elementCount();

    var count: usize = 0;
    var i: usize = 0;
    var j: usize = 0;

    while (i < n_a and j < n_b) {
        const val_a = try u_a.values.getAsFloat(&.{i});
        const val_b = try u_b.values.getAsFloat(&.{j});
        if (val_a == val_b) {
            count += 1;
            i += 1;
            j += 1;
        } else if (val_a < val_b) {
            i += 1;
        } else {
            j += 1;
        }
    }

    var out = try empty(a.allocator, .{ .shape = &.{count}, .dtype = a.dtype });
    errdefer out.deinit();

    i = 0;
    j = 0;
    var write_idx: usize = 0;
    while (i < n_a and j < n_b) {
        const val_a = try u_a.values.getAsFloat(&.{i});
        const val_b = try u_b.values.getAsFloat(&.{j});
        if (val_a == val_b) {
            try out.setFromFloat(&.{write_idx}, val_a);
            write_idx += 1;
            i += 1;
            j += 1;
        } else if (val_a < val_b) {
            i += 1;
        } else {
            j += 1;
        }
    }

    return out;
}

/// Find the sorted, unique union of two arrays.
pub fn union1d(
    a: Array,
    b: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const concat = @import("../manip/concat.zig").concat;
    var flat_a = try ravel(a);
    defer flat_a.deinit();
    var flat_b = try ravel(b);
    defer flat_b.deinit();

    var cat = try concat(&.{ flat_a, flat_b }, .{ .axis = 0 });
    defer cat.deinit();

    const res = try unique(cat, .{});
    return res.values;
}

/// Find the set difference of two arrays: elements in `a` that are not in `b`.
pub fn setdiff1d(
    a: Array,
    b: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var u_a = try unique(a, .{});
    defer u_a.deinit();
    var u_b = try unique(b, .{});
    defer u_b.deinit();

    const n_a = u_a.values.elementCount();
    const n_b = u_b.values.elementCount();

    var count: usize = 0;
    for (0..n_a) |i| {
        const val_a = try u_a.values.getAsFloat(&.{i});
        var found = false;
        for (0..n_b) |j| {
            if (val_a == try u_b.values.getAsFloat(&.{j})) {
                found = true;
                break;
            }
        }
        if (!found) count += 1;
    }

    var out = try empty(a.allocator, .{ .shape = &.{count}, .dtype = a.dtype });
    errdefer out.deinit();

    var write_idx: usize = 0;
    for (0..n_a) |i| {
        const val_a = try u_a.values.getAsFloat(&.{i});
        var found = false;
        for (0..n_b) |j| {
            if (val_a == try u_b.values.getAsFloat(&.{j})) {
                found = true;
                break;
            }
        }
        if (!found) {
            try out.setFromFloat(&.{write_idx}, val_a);
            write_idx += 1;
        }
    }

    return out;
}

/// Calculates element in `element`, testing membership in `test_elements`.
pub fn isin(
    element: Array,
    test_elements: Array,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    var out = try empty(element.allocator, .{ .shape = element.shapeSlice(), .dtype = .bool });
    errdefer out.deinit();

    const n_test = test_elements.elementCount();
    var flat_test = try ravel(test_elements);
    defer flat_test.deinit();

    const test_buf = try element.allocator.alloc(f64, n_test);
    defer element.allocator.free(test_buf);
    for (0..n_test) |k| {
        test_buf[k] = try flat_test.getAsFloat(&.{k});
    }

    var it = @import("../core/iterator.zig").NdIterator.init(element.shape(), element.strides());
    var out_it = @import("../core/iterator.zig").NdIterator.init(out.shape(), out.strides());

    while (it.next()) |item| {
        const out_item = out_it.next().?;
        const v = try element.getAsFloat(item.indices[0..element.ndim]);
        var matched = false;
        for (test_buf) |tb| {
            if (v == tb) {
                matched = true;
                break;
            }
        }
        const out_ptr: [*]bool = @ptrCast(@alignCast(out.data_ptr));
        out_ptr[@as(usize, @intCast(out_item.offset))] = matched;
    }

    return out;
}

test "searchSorted binary insertion" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const s_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var s_arr = try fromSlice(allocator, f64, .{ .data = &s_data, .shape = &.{5} });
    defer s_arr.deinit();

    const q_data = [_]f64{ 2.5, 0.0, 5.0 };
    var q_arr = try fromSlice(allocator, f64, .{ .data = &q_data, .shape = &.{3} });
    defer q_arr.deinit();

    var idx_left = try searchSorted(s_arr, q_arr, .{ .side = .left });
    defer idx_left.deinit();

    try testing.expectEqual(@as(i64, 2), try idx_left.get(i64, &.{0}));
    try testing.expectEqual(@as(i64, 0), try idx_left.get(i64, &.{1}));
    try testing.expectEqual(@as(i64, 4), try idx_left.get(i64, &.{2}));

    var idx_right = try searchSorted(s_arr, q_arr, .{ .side = .right });
    defer idx_right.deinit();

    try testing.expectEqual(@as(i64, 2), try idx_right.get(i64, &.{0}));
    try testing.expectEqual(@as(i64, 0), try idx_right.get(i64, &.{1}));
    try testing.expectEqual(@as(i64, 5), try idx_right.get(i64, &.{2}));
}

test "unique deduplication" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]f64{ 3.0, 1.0, 2.0, 3.0, 1.0, 4.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{6} });
    defer arr.deinit();

    const res = try unique(arr, .{ .returnIndex = true, .returnCounts = true });
    var u_arr = res.values;
    defer u_arr.deinit();

    var idx_arr = res.indices.?;
    defer idx_arr.deinit();

    var cnt_arr = res.counts.?;
    defer cnt_arr.deinit();

    try testing.expectEqual(@as(usize, 4), u_arr.elementCount());
    try testing.expectEqual(@as(f64, 1.0), try u_arr.get(f64, &.{0}));
    try testing.expectEqual(@as(f64, 2.0), try u_arr.get(f64, &.{1}));
    try testing.expectEqual(@as(f64, 3.0), try u_arr.get(f64, &.{2}));
    try testing.expectEqual(@as(f64, 4.0), try u_arr.get(f64, &.{3}));

    // counts for 1.0 (2), 2.0 (1), 3.0 (2), 4.0 (1)
    try testing.expectEqual(@as(i64, 2), try cnt_arr.get(i64, &.{0}));
    try testing.expectEqual(@as(i64, 1), try cnt_arr.get(i64, &.{1}));
    try testing.expectEqual(@as(i64, 2), try cnt_arr.get(i64, &.{2}));
    try testing.expectEqual(@as(i64, 1), try cnt_arr.get(i64, &.{3}));
}

test "flatNonzero" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const data = [_]f64{ 0.0, 1.0, 0.0, 2.5, 0.0, 9.9 };
    var arr = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer arr.deinit();

    var fnz = try flatNonzero(arr);
    defer fnz.deinit();

    try testing.expectEqual(@as(usize, 3), fnz.elementCount());
    try testing.expectEqual(@as(i64, 1), try fnz.get(i64, &.{0}));
    try testing.expectEqual(@as(i64, 3), try fnz.get(i64, &.{1}));
    try testing.expectEqual(@as(i64, 5), try fnz.get(i64, &.{2}));

    // intersect1d and union1d
    const set_a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const set_b = [_]f64{ 3.0, 4.0, 5.0, 6.0 };
    var sa = try fromSlice(allocator, f64, .{ .data = &set_a, .shape = &.{4} });
    defer sa.deinit();
    var sb = try fromSlice(allocator, f64, .{ .data = &set_b, .shape = &.{4} });
    defer sb.deinit();

    var inter = try intersect1d(sa, sb);
    defer inter.deinit();
    try testing.expectEqualSlices(f64, &.{ 3.0, 4.0 }, try inter.asSlice(f64));

    var un = try union1d(sa, sb);
    defer un.deinit();
    try testing.expectEqualSlices(f64, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, try un.asSlice(f64));

    var diff_set = try setdiff1d(sa, sb);
    defer diff_set.deinit();
    try testing.expectEqualSlices(f64, &.{ 1.0, 2.0 }, try diff_set.asSlice(f64));

    var isin_mask = try isin(sa, sb);
    defer isin_mask.deinit();
    try testing.expectEqualSlices(bool, &.{ false, false, true, true }, try isin_mask.asSlice(bool));

    // argwhere / nonzero
    var aw = try argwhere(arr);
    defer aw.deinit();
    try testing.expectEqualSlices(usize, &.{ 3, 2 }, aw.shapeSlice());
    // row 0, col 1
    try testing.expectEqual(@as(i64, 0), try aw.get(i64, &.{ 0, 0 }));
    try testing.expectEqual(@as(i64, 1), try aw.get(i64, &.{ 0, 1 }));
    // row 1, col 0
    try testing.expectEqual(@as(i64, 1), try aw.get(i64, &.{ 1, 0 }));
    try testing.expectEqual(@as(i64, 0), try aw.get(i64, &.{ 1, 1 }));
    // row 1, col 2
    try testing.expectEqual(@as(i64, 1), try aw.get(i64, &.{ 2, 0 }));
    try testing.expectEqual(@as(i64, 2), try aw.get(i64, &.{ 2, 1 }));
}
