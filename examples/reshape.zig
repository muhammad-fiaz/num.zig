//! Demonstrates shape transformations: reshape, expandDims, squeeze, ravel, and flatten with f32 and f64 dtypes.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. f32 array shape transformations
    var arr = try num.arange(allocator, .{
        .start = 0,
        .stop = 12,
        .step = 1,
        .dtype = .f32,
    });
    defer arr.deinit();

    var mat = try num.manip.reshape(arr, .{
        .shape = &.{ 3, 4 },
    });
    defer mat.deinit();

    const value = try mat.get(f32, &.{ 2, 3 });
    std.debug.print(
        "Reshaped shape: [{d}, {d}], value: {d:.1}\n",
        .{
            mat.shape_dims[0],
            mat.shape_dims[1],
            value,
        },
    );

    var expanded = try num.manip.expandDims(mat, .{
        .axis = 1,
    });
    defer expanded.deinit();
    std.debug.print(
        "Expanded shape (axis 1): [{d}, {d}, {d}]\n",
        .{
            expanded.shape_dims[0],
            expanded.shape_dims[1],
            expanded.shape_dims[2],
        },
    );

    var squeezed = try num.manip.squeeze(expanded, .{
        .axis = 1,
    });
    defer squeezed.deinit();
    std.debug.print("Squeezed shape (axis 1): [{d}, {d}]\n", .{
        squeezed.shape_dims[0],
        squeezed.shape_dims[1],
    });

    var flat = try num.manip.ravel(mat);
    defer flat.deinit();

    const last = try flat.get(f32, &.{11});
    std.debug.print(
        "Raveled length: {d}, last value: {d:.1}\n",
        .{
            flat.elementCount(),
            last,
        },
    );

    // 2. f64 array shape transformations and flatten
    var arr_f64 = try num.arange(allocator, .{
        .start = 0,
        .stop = 6,
        .step = 1,
        .dtype = .f64,
    });
    defer arr_f64.deinit();

    var mat_f64 = try num.manip.reshape(arr_f64, .{
        .shape = &.{ 2, 3 },
    });
    defer mat_f64.deinit();

    var flattened_f64 = try num.manip.flatten(mat_f64);
    defer flattened_f64.deinit();

    std.debug.print(
        "Flattened f64 length: {d}, val[4]: {d:.1}\n",
        .{
            flattened_f64.elementCount(),
            try flattened_f64.get(f64, &.{4}),
        },
    );
}
