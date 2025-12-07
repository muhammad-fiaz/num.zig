const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Performs a generic binary operation with broadcasting.
fn binaryOpGeneric(
    allocator: Allocator,
    comptime T: type,
    comptime R: type,
    a: NDArray(T),
    b: NDArray(T),
    comptime op: anytype,
) !NDArray(R) {
    const shape = try core.broadcastShape(allocator, a.shape, b.shape);
    defer allocator.free(shape);

    var a_broad = try a.broadcastTo(shape);
    defer a_broad.deinit();

    var b_broad = try b.broadcastTo(shape);
    defer b_broad.deinit();

    var result = try NDArray(R).init(allocator, shape);
    errdefer result.deinit();

    var iter = try core.NdIterator.init(allocator, shape);
    defer iter.deinit();

    var i: usize = 0;
    while (iter.next()) |coords| {
        const val_a = try a_broad.get(coords);
        const val_b = try b_broad.get(coords);
        result.data[i] = op(val_a, val_b);
        i += 1;
    }

    return result;
}

fn binaryOp(
    allocator: Allocator,
    comptime T: type,
    a: NDArray(T),
    b: NDArray(T),
    comptime op: anytype,
) !NDArray(T) {
    return binaryOpGeneric(allocator, T, T, a, b, op);
}

fn add_impl(comptime T: type) fn (T, T) T {
    return struct {
        fn impl(a: T, b: T) T {
            switch (@typeInfo(T)) {
                .@"struct" => {
                    if (@hasDecl(T, "add")) {
                        return T.add(a, b);
                    } else {
                        @compileError("Struct type does not have add method");
                    }
                },
                else => return a + b,
            }
        }
    }.impl;
}
fn sub_impl(comptime T: type) fn (T, T) T {
    return struct {
        fn impl(a: T, b: T) T {
            switch (@typeInfo(T)) {
                .@"struct" => {
                    if (@hasDecl(T, "sub")) {
                        return T.sub(a, b);
                    } else {
                        @compileError("Struct type does not have sub method");
                    }
                },
                else => return a - b,
            }
        }
    }.impl;
}
fn mul_impl(comptime T: type) fn (T, T) T {
    return struct {
        fn impl(a: T, b: T) T {
            switch (@typeInfo(T)) {
                .@"struct" => {
                    if (@hasDecl(T, "mul")) {
                        return T.mul(a, b);
                    } else {
                        @compileError("Struct type does not have mul method");
                    }
                },
                else => return a * b,
            }
        }
    }.impl;
}
fn div_impl(comptime T: type) fn (T, T) T {
    return struct {
        fn impl(a: T, b: T) T {
            switch (@typeInfo(T)) {
                .@"struct" => {
                    if (@hasDecl(T, "div")) {
                        return T.div(a, b);
                    } else {
                        @compileError("Struct type does not have div method");
                    }
                },
                else => return a / b,
            }
        }
    }.impl;
}

/// Performs element-wise addition of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the sum of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 4.0});
/// defer b.deinit();
///
/// var result = try elementwise.add(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {4.0, 6.0}
/// ```
pub fn add(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, add_impl(T));
}

/// Performs element-wise subtraction of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the difference of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{5.0, 6.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.sub(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {3.0, 3.0}
/// ```
pub fn sub(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, sub_impl(T));
}

/// Performs element-wise multiplication of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the product of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{4.0, 5.0});
/// defer b.deinit();
///
/// var result = try elementwise.mul(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {8.0, 15.0}
/// ```
pub fn mul(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, mul_impl(T));
}

/// Performs element-wise division of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the quotient of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{10.0, 20.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 4.0});
/// defer b.deinit();
///
/// var result = try elementwise.div(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {5.0, 5.0}
/// ```
pub fn div(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, div_impl(T));
}

fn pow_impl(a: anytype, b: anytype) @TypeOf(a) {
    return std.math.pow(@TypeOf(a), a, b);
}

/// Performs element-wise power of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The base array.
///     b: The exponent array.
///
/// Returns:
///     A new array containing the bases raised to the exponents.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{3.0, 2.0});
/// defer b.deinit();
///
/// var result = try elementwise.pow(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {8.0, 9.0}
/// ```
pub fn pow(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, pow_impl);
}

/// Generic unary operation.
fn unaryOpGeneric(
    allocator: Allocator,
    comptime T: type,
    comptime R: type,
    a: NDArray(T),
    comptime op: anytype,
) !NDArray(R) {
    var result = try NDArray(R).init(allocator, a.shape);

    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    var i: usize = 0;
    while (iter.next()) |coords| {
        const val = try a.get(coords);
        result.data[i] = op(val);
        i += 1;
    }
    return result;
}

fn unaryOp(
    allocator: Allocator,
    comptime T: type,
    a: NDArray(T),
    comptime op: anytype,
) !NDArray(T) {
    return unaryOpGeneric(allocator, T, T, a, op);
}

fn exp_impl(val: anytype) @TypeOf(val) {
    return @exp(val);
}
fn log_impl(val: anytype) @TypeOf(val) {
    return @log(val);
}
fn sqrt_impl(val: anytype) @TypeOf(val) {
    return @sqrt(val);
}
fn sin_impl(val: anytype) @TypeOf(val) {
    return @sin(val);
}
fn cos_impl(val: anytype) @TypeOf(val) {
    return @cos(val);
}

pub fn exp(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, exp_impl);
}

/// Computes the natural logarithm of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the natural logarithm of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.71828});
/// defer a.deinit();
///
/// var result = try elementwise.log(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 1.0}
/// ```
pub fn log(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, log_impl);
}

/// Computes the square root of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the square root of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{4.0, 9.0});
/// defer a.deinit();
///
/// var result = try elementwise.sqrt(allocator, f32, a);
/// defer result.deinit();
/// // result is {2.0, 3.0}
/// ```
pub fn sqrt(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, sqrt_impl);
}

/// Computes the sine of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the sine of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 1.5708});
/// defer a.deinit();
///
/// var result = try elementwise.sin(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 1.0}
/// ```
pub fn sin(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, sin_impl);
}

/// Computes the cosine of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the cosine of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 3.14159});
/// defer a.deinit();
///
/// var result = try elementwise.cos(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {1.0, -1.0}
/// ```
pub fn cos(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, cos_impl);
}

fn tan_impl(val: anytype) @TypeOf(val) {
    return std.math.tan(val);
}
fn asin_impl(val: anytype) @TypeOf(val) {
    return std.math.asin(val);
}
fn acos_impl(val: anytype) @TypeOf(val) {
    return std.math.acos(val);
}
fn atan_impl(val: anytype) @TypeOf(val) {
    return std.math.atan(val);
}

/// Computes the tangent of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the tangent of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 0.7854});
/// defer a.deinit();
///
/// var result = try elementwise.tan(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 1.0}
/// ```
pub fn tan(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, tan_impl);
}

/// Computes the arc sine of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the arc sine of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 1.0});
/// defer a.deinit();
///
/// var result = try elementwise.arcsin(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 1.5708}
/// ```
pub fn arcsin(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, asin_impl);
}

/// Computes the arc cosine of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the arc cosine of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 0.0});
/// defer a.deinit();
///
/// var result = try elementwise.arccos(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 1.5708}
/// ```
pub fn arccos(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, acos_impl);
}

/// Computes the arc tangent of each element.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the array.
///     a: The input array.
///
/// Returns:
///     A new array containing the arc tangent of the input elements.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{0.0, 1.0});
/// defer a.deinit();
///
/// var result = try elementwise.arctan(allocator, f32, a);
/// defer result.deinit();
/// // result is approx {0.0, 0.7854}
/// ```
pub fn arctan(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, atan_impl);
}

// Logical
fn logical_and_impl(a: bool, b: bool) bool {
    return a and b;
}
fn logical_or_impl(a: bool, b: bool) bool {
    return a or b;
}
fn logical_xor_impl(a: bool, b: bool) bool {
    return a != b;
}
fn logical_not_impl(a: bool) bool {
    return !a;
}

/// Performs element-wise logical AND of two boolean arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array containing the logical AND of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(bool).init(allocator, &.{2}, &.{true, false});
/// defer a.deinit();
/// var b = try NDArray(bool).init(allocator, &.{2}, &.{true, true});
/// defer b.deinit();
///
/// var result = try elementwise.logical_and(allocator, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn logical_and(allocator: Allocator, a: NDArray(bool), b: NDArray(bool)) !NDArray(bool) {
    return binaryOpGeneric(allocator, bool, bool, a, b, logical_and_impl);
}

/// Performs element-wise logical OR of two boolean arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array containing the logical OR of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(bool).init(allocator, &.{2}, &.{true, false});
/// defer a.deinit();
/// var b = try NDArray(bool).init(allocator, &.{2}, &.{false, false});
/// defer b.deinit();
///
/// var result = try elementwise.logical_or(allocator, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn logical_or(allocator: Allocator, a: NDArray(bool), b: NDArray(bool)) !NDArray(bool) {
    return binaryOpGeneric(allocator, bool, bool, a, b, logical_or_impl);
}

/// Performs element-wise logical XOR of two boolean arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array containing the logical XOR of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(bool).init(allocator, &.{2}, &.{true, false});
/// defer a.deinit();
/// var b = try NDArray(bool).init(allocator, &.{2}, &.{true, true});
/// defer b.deinit();
///
/// var result = try elementwise.logical_xor(allocator, a, b);
/// defer result.deinit();
/// // result is {false, true}
/// ```
pub fn logical_xor(allocator: Allocator, a: NDArray(bool), b: NDArray(bool)) !NDArray(bool) {
    return binaryOpGeneric(allocator, bool, bool, a, b, logical_xor_impl);
}

/// Performs element-wise logical NOT of a boolean array.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     a: The input array.
///
/// Returns:
///     A new boolean array containing the logical NOT of the input array.
///
/// Example:
/// ```zig
/// var a = try NDArray(bool).init(allocator, &.{2}, &.{true, false});
/// defer a.deinit();
///
/// var result = try elementwise.logical_not(allocator, a);
/// defer result.deinit();
/// // result is {false, true}
/// ```
pub fn logical_not(allocator: Allocator, a: NDArray(bool)) !NDArray(bool) {
    return unaryOpGeneric(allocator, bool, bool, a, logical_not_impl);
}

// Bitwise (integers only)
fn bitwise_and_impl(a: anytype, b: anytype) @TypeOf(a) {
    return a & b;
}
fn bitwise_or_impl(a: anytype, b: anytype) @TypeOf(a) {
    return a | b;
}
fn bitwise_xor_impl(a: anytype, b: anytype) @TypeOf(a) {
    return a ^ b;
}

/// Performs element-wise bitwise AND of two integer arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays (must be integer).
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the bitwise AND of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(i32).init(allocator, &.{2}, &.{5, 3});
/// defer a.deinit();
/// var b = try NDArray(i32).init(allocator, &.{2}, &.{3, 1});
/// defer b.deinit();
///
/// var result = try elementwise.bitwise_and(allocator, i32, a, b);
/// defer result.deinit();
/// // result is {1, 1}
/// ```
pub fn bitwise_and(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, bitwise_and_impl);
}

/// Performs element-wise bitwise OR of two integer arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays (must be integer).
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the bitwise OR of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(i32).init(allocator, &.{2}, &.{5, 3});
/// defer a.deinit();
/// var b = try NDArray(i32).init(allocator, &.{2}, &.{3, 1});
/// defer b.deinit();
///
/// var result = try elementwise.bitwise_or(allocator, i32, a, b);
/// defer result.deinit();
/// // result is {7, 3}
/// ```
pub fn bitwise_or(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, bitwise_or_impl);
}

/// Performs element-wise bitwise XOR of two integer arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays (must be integer).
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new array containing the bitwise XOR of the input arrays.
///
/// Example:
/// ```zig
/// var a = try NDArray(i32).init(allocator, &.{2}, &.{5, 3});
/// defer a.deinit();
/// var b = try NDArray(i32).init(allocator, &.{2}, &.{3, 1});
/// defer b.deinit();
///
/// var result = try elementwise.bitwise_xor(allocator, i32, a, b);
/// defer result.deinit();
/// // result is {6, 2}
/// ```
pub fn bitwise_xor(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, bitwise_xor_impl);
}

// Comparison
fn eq_impl(a: anytype, b: anytype) bool {
    return a == b;
}
fn neq_impl(a: anytype, b: anytype) bool {
    return a != b;
}
fn gt_impl(a: anytype, b: anytype) bool {
    return a > b;
}
fn ge_impl(a: anytype, b: anytype) bool {
    return a >= b;
}
fn lt_impl(a: anytype, b: anytype) bool {
    return a < b;
}
fn le_impl(a: anytype, b: anytype) bool {
    return a <= b;
}

/// Performs element-wise equality comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates equality.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.equal(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn equal(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, eq_impl);
}

/// Performs element-wise inequality comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates inequality.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.not_equal(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {false, true}
/// ```
pub fn not_equal(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, neq_impl);
}

/// Performs element-wise greater-than comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates a > b.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.greater(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn greater(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, gt_impl);
}

/// Performs element-wise greater-than-or-equal comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates a >= b.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 2.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.greater_equal(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn greater_equal(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, ge_impl);
}

/// Performs element-wise less-than comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates a < b.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{1.0, 4.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.less(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn less(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, lt_impl);
}

/// Performs element-wise less-than-or-equal comparison of two arrays with broadcasting.
///
/// Arguments:
///     allocator: The allocator to use for the result array.
///     T: The type of elements in the arrays.
///     a: The first input array.
///     b: The second input array.
///
/// Returns:
///     A new boolean array where true indicates a <= b.
///
/// Example:
/// ```zig
/// var a = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 4.0});
/// defer a.deinit();
/// var b = try NDArray(f32).init(allocator, &.{2}, &.{2.0, 3.0});
/// defer b.deinit();
///
/// var result = try elementwise.less_equal(allocator, f32, a, b);
/// defer result.deinit();
/// // result is {true, false}
/// ```
pub fn less_equal(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(bool) {
    return binaryOpGeneric(allocator, T, bool, a, b, le_impl);
}

fn neg_impl(val: anytype) @TypeOf(val) {
    return -val;
}

/// Computes the negation of each element.
pub fn neg(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, neg_impl);
}

fn abs_impl(val: anytype) @TypeOf(val) {
    return @abs(val);
}

/// Computes the absolute value of each element.
pub fn abs(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, abs_impl);
}

fn round_impl(val: anytype) @TypeOf(val) {
    return @round(val);
}

/// Rounds each element to the nearest integer.
pub fn round(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, round_impl);
}

fn floor_impl(val: anytype) @TypeOf(val) {
    return @floor(val);
}

/// Rounds each element down to the nearest integer.
pub fn floor(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, floor_impl);
}

fn ceil_impl(val: anytype) @TypeOf(val) {
    return @ceil(val);
}

/// Rounds each element up to the nearest integer.
pub fn ceil(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, ceil_impl);
}

fn sign_impl(val: anytype) @TypeOf(val) {
    if (val > 0) return 1;
    if (val < 0) return -1;
    return 0;
}

/// Returns the sign of each element (-1, 0, or 1).
pub fn sign(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T) {
    return unaryOp(allocator, T, a, sign_impl);
}

fn max_impl(a: anytype, b: anytype) @TypeOf(a) {
    return @max(a, b);
}

/// Element-wise maximum of two arrays.
pub fn maximum(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, max_impl);
}

fn min_impl(a: anytype, b: anytype) @TypeOf(a) {
    return @min(a, b);
}

/// Element-wise minimum of two arrays.
pub fn minimum(allocator: Allocator, comptime T: type, a: NDArray(T), b: NDArray(T)) !NDArray(T) {
    return binaryOp(allocator, T, a, b, min_impl);
}

/// Clip (limit) the values in an array.
/// Given an interval, values outside the interval are clipped to the interval edges.
pub fn clip(allocator: Allocator, comptime T: type, a: NDArray(T), min_val: T, max_val: T) !NDArray(T) {
    var result = try NDArray(T).init(allocator, a.shape);

    var iter = try core.NdIterator.init(allocator, a.shape);
    defer iter.deinit();

    var i: usize = 0;
    while (iter.next()) |coords| {
        const val = try a.get(coords);
        if (val < min_val) {
            result.data[i] = min_val;
        } else if (val > max_val) {
            result.data[i] = max_val;
        } else {
            result.data[i] = val;
        }
        i += 1;
    }
    return result;
}

test "elementwise add broadcasting" {
    const allocator = std.testing.allocator;

    var a = try NDArray(f32).ones(allocator, &.{ 2, 3 });
    defer a.deinit();

    var b = try NDArray(f32).full(allocator, &.{ 1, 3 }, 2.0);
    defer b.deinit();

    var c = try add(allocator, f32, a, b);
    defer c.deinit();

    try std.testing.expectEqual(c.shape[0], 2);
    try std.testing.expectEqual(c.shape[1], 3);
    try std.testing.expectEqual(try c.get(&.{ 0, 0 }), 3.0);
    try std.testing.expectEqual(try c.get(&.{ 1, 2 }), 3.0);
}

test "elementwise complex broadcasting" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).ones(allocator, &.{ 1, 3, 1 });
    defer a.deinit();
    var b = try NDArray(f32).full(allocator, &.{ 2, 1, 4 }, 2.0);
    defer b.deinit();

    var c = try add(allocator, f32, a, b);
    defer c.deinit();

    try std.testing.expectEqual(c.shape[0], 2);
    try std.testing.expectEqual(c.shape[1], 3);
    try std.testing.expectEqual(c.shape[2], 4);
    try std.testing.expectEqual(try c.get(&.{ 0, 0, 0 }), 3.0);
}
