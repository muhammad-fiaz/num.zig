const std = @import("std");
const core = @import("core.zig");
const broadcast = @import("broadcast.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

fn elementwise(
    comptime T: type,
    allocator: Allocator,
    a: *const NDArray(T),
    b: *const NDArray(T),
    comptime op: anytype,
) !NDArray(T) {
    const out_shape = try broadcast.broadcastShape(allocator, a.shape, b.shape);
    defer allocator.free(out_shape);

    var result = try NDArray(T).init(allocator, out_shape);
    errdefer result.deinit();

    const total_elements = result.size();
    const rank = result.rank();

    // Optimization for same shape (no broadcasting needed)
    if (std.mem.eql(usize, a.shape, b.shape)) {
        for (result.data, 0..) |*r, i| {
            r.* = op(a.data[i], b.data[i]);
        }
        return result;
    }

    const a_strides_broadcast = try broadcast.broadcastStrides(allocator, a.shape, a.strides, out_shape);
    defer allocator.free(a_strides_broadcast);
    const b_strides_broadcast = try broadcast.broadcastStrides(allocator, b.shape, b.strides, out_shape);
    defer allocator.free(b_strides_broadcast);

    var coords = try allocator.alloc(usize, rank);
    defer allocator.free(coords);
    @memset(coords, 0);

    var offset: usize = 0;
    while (offset < total_elements) : (offset += 1) {
        var offset_a: usize = 0;
        var offset_b: usize = 0;

        for (coords, 0..) |c, dim| {
            offset_a += c * a_strides_broadcast[dim];
            offset_b += c * b_strides_broadcast[dim];
        }

        result.data[offset] = op(a.data[offset_a], b.data[offset_b]);

        // Increment coords
        if (rank > 0) {
            var dim = rank - 1;
            while (true) {
                coords[dim] += 1;
                if (coords[dim] < result.shape[dim]) break;
                coords[dim] = 0;
                if (dim == 0) break;
                dim -= 1;
            }
        }
    }

    return result;
}

fn unaryElementwise(
    comptime T: type,
    allocator: Allocator,
    a: *const NDArray(T),
    comptime op: anytype,
) !NDArray(T) {
    const result = try NDArray(T).init(allocator, a.shape);
    for (result.data, 0..) |*r, i| {
        r.* = op(a.data[i]);
    }
    return result;
}

fn add_op(a: anytype, b: anytype) @TypeOf(a) {
    return a + b;
}
fn sub_op(a: anytype, b: anytype) @TypeOf(a) {
    return a - b;
}
fn mul_op(a: anytype, b: anytype) @TypeOf(a) {
    return a * b;
}
fn div_op(a: anytype, b: anytype) @TypeOf(a) {
    return a / b;
}
fn pow_op(a: anytype, b: anytype) @TypeOf(a) {
    return std.math.pow(@TypeOf(a), a, b);
}
fn eq_op(a: anytype, b: anytype) bool {
    return a == b;
} // Returns bool, need to handle return type T?
// elementwise expects op to return T.
// If we want boolean arrays, we need NDArray(bool).
// But elementwise assumes T.
// So we need a separate elementwise for comparisons or change elementwise signature.
// For now, let's stick to math ops returning T.

fn exp_op(val: anytype) @TypeOf(val) {
    return std.math.exp(val);
}
fn log_op(val: anytype) @TypeOf(val) {
    return std.math.ln(val);
}
fn sqrt_op(val: anytype) @TypeOf(val) {
    return std.math.sqrt(val);
}
fn abs_op(val: anytype) @TypeOf(val) {
    return @abs(val);
}
fn neg_op(val: anytype) @TypeOf(val) {
    return -val;
}

/// Elementwise addition with broadcasting.
///
/// Computes `a + b` element-wise. Supports broadcasting if shapes are compatible.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The first input NDArray.
///     b: The second input NDArray.
///
/// Returns:
///     A new NDArray containing the result of the addition.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 4.0});
/// defer b.deinit();
///
/// var result = try ops.add(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {4.0, 6.0}
/// ```
pub fn add(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    return elementwise(T, allocator, a, b, add_op);
}

/// Elementwise subtraction with broadcasting.
///
/// Computes `a - b` element-wise. Supports broadcasting if shapes are compatible.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The first input NDArray.
///     b: The second input NDArray.
///
/// Returns:
///     A new NDArray containing the result of the subtraction.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{5.0, 7.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer b.deinit();
///
/// var result = try ops.sub(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {3.0, 4.0}
/// ```
pub fn sub(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    return elementwise(T, allocator, a, b, sub_op);
}

/// Elementwise multiplication with broadcasting.
///
/// Computes `a * b` element-wise. Supports broadcasting if shapes are compatible.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The first input NDArray.
///     b: The second input NDArray.
///
/// Returns:
///     A new NDArray containing the result of the multiplication.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{4.0, 5.0});
/// defer b.deinit();
///
/// var result = try ops.mul(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {8.0, 15.0}
/// ```
pub fn mul(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    return elementwise(T, allocator, a, b, mul_op);
}

/// Elementwise division with broadcasting.
///
/// Computes `a / b` element-wise. Supports broadcasting if shapes are compatible.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The first input NDArray.
///     b: The second input NDArray.
///
/// Returns:
///     A new NDArray containing the result of the division.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{10.0, 20.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 4.0});
/// defer b.deinit();
///
/// var result = try ops.div(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {5.0, 5.0}
/// ```
pub fn div(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    return elementwise(T, allocator, a, b, div_op);
}

/// Elementwise power with broadcasting.
///
/// Computes `a ^ b` element-wise. Supports broadcasting if shapes are compatible.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The base NDArray.
///     b: The exponent NDArray.
///
/// Returns:
///     A new NDArray containing the result of the power operation.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 2.0});
/// defer b.deinit();
///
/// var result = try ops.pow(f32, allocator, &a, &b);
/// defer result.deinit();
/// // result is {8.0, 9.0}
/// ```
pub fn pow(comptime T: type, allocator: Allocator, a: *const NDArray(T), b: *const NDArray(T)) !NDArray(T) {
    return elementwise(T, allocator, a, b, pow_op);
}

/// Elementwise exponential.
///
/// Computes `e ^ x` for each element `x` in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///
/// Returns:
///     A new NDArray containing the exponential values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 1.0});
/// defer a.deinit();
///
/// var result = try ops.exp(f32, allocator, &a);
/// defer result.deinit();
/// // result is {1.0, 2.718...}
/// ```
pub fn exp(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    return unaryElementwise(T, allocator, a, exp_op);
}

/// Elementwise natural logarithm.
///
/// Computes `ln(x)` for each element `x` in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///
/// Returns:
///     A new NDArray containing the natural logarithm values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.71828});
/// defer a.deinit();
///
/// var result = try ops.log(f32, allocator, &a);
/// defer result.deinit();
/// // result is {0.0, 1.0}
/// ```
pub fn log(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    return unaryElementwise(T, allocator, a, log_op);
}

/// Elementwise square root.
///
/// Computes `sqrt(x)` for each element `x` in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///
/// Returns:
///     A new NDArray containing the square root values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{4.0, 9.0});
/// defer a.deinit();
///
/// var result = try ops.sqrt(f32, allocator, &a);
/// defer result.deinit();
/// // result is {2.0, 3.0}
/// ```
pub fn sqrt(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    return unaryElementwise(T, allocator, a, sqrt_op);
}

/// Elementwise absolute value.
///
/// Computes `|x|` for each element `x` in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///
/// Returns:
///     A new NDArray containing the absolute values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{ -1.0, 2.0 });
/// defer a.deinit();
///
/// var result = try ops.abs(f32, allocator, &a);
/// defer result.deinit();
/// // result is {1.0, 2.0}
/// ```
pub fn abs(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    return unaryElementwise(T, allocator, a, abs_op);
}

/// Elementwise negation.
///
/// Computes `-x` for each element `x` in the array.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///
/// Returns:
///     A new NDArray containing the negated values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, -2.0});
/// defer a.deinit();
///
/// var result = try ops.neg(f32, allocator, &a);
/// defer result.deinit();
/// // result is {-1.0, 2.0}
/// ```
pub fn neg(comptime T: type, allocator: Allocator, a: *const NDArray(T)) !NDArray(T) {
    return unaryElementwise(T, allocator, a, neg_op);
}

/// Clips the values in the array to a specified range.
///
/// Values smaller than `min_val` are set to `min_val`, and values larger than `max_val` are set to `max_val`.
///
/// Arguments:
///     T: The data type of the array elements.
///     allocator: The allocator to use for the result array.
///     a: The input NDArray.
///     min_val: The minimum value.
///     max_val: The maximum value.
///
/// Returns:
///     A new NDArray containing the clipped values.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{3}, &.{0.0, 5.0, 10.0});
/// defer a.deinit();
///
/// var result = try ops.clip(f32, allocator, &a, 2.0, 8.0);
/// defer result.deinit();
/// // result is {2.0, 5.0, 8.0}
/// ```
pub fn clip(comptime T: type, allocator: Allocator, a: *const NDArray(T), min_val: T, max_val: T) !NDArray(T) {
    const result = try NDArray(T).init(allocator, a.shape);
    for (a.data, 0..) |val, i| {
        if (val < min_val) {
            result.data[i] = min_val;
        } else if (val > max_val) {
            result.data[i] = max_val;
        } else {
            result.data[i] = val;
        }
    }
    return result;
}
