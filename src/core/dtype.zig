//! Data types, size/alignment metadata, and promotion rules.
//!
//! Defines the core DType enumeration representing all supported numerical types,
//! compile-time mapping between Zig types and runtime tags, and standard type promotion.

const std = @import("std");
const DTypeError = @import("error.zig").DTypeError;

/// Enumeration of all supported numerical scalar types.
pub const DType = enum(u8) {
    bool = 0,
    i8 = 1,
    i16 = 2,
    i32 = 3,
    i64 = 4,
    u8 = 5,
    u16 = 6,
    u32 = 7,
    u64 = 8,
    f16 = 9,
    f32 = 10,
    f64 = 11,
    c64 = 12,
    c128 = 13,

    /// Maps a runtime DType to its corresponding Zig native type at compile time.
    pub fn toType(comptime self: DType) type {
        return switch (self) {
            .bool => bool,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
            .f16 => f16,
            .f32 => f32,
            .f64 => f64,
            .c64 => std.math.Complex(f32),
            .c128 => std.math.Complex(f64),
        };
    }

    /// Derives the DType enum from a native Zig type at compile time.
    pub fn fromType(comptime T: type) DType {
        return switch (T) {
            bool => .bool,
            i8 => .i8,
            i16 => .i16,
            i32 => .i32,
            i64 => .i64,
            u8 => .u8,
            u16 => .u16,
            u32 => .u32,
            u64 => .u64,
            f16 => .f16,
            f32 => .f32,
            f64 => .f64,
            std.math.Complex(f32) => .c64,
            std.math.Complex(f64) => .c128,
            else => @compileError("Unsupported array element type: " ++ @typeName(T)),
        };
    }

    /// Size in bytes of a single scalar element of this data type.
    pub fn sizeOf(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => 1,
            .i16, .u16, .f16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64, .c64 => 8,
            .c128 => 16,
        };
    }

    /// Memory alignment in bytes of a single scalar element of this data type.
    pub fn alignmentOf(self: DType) usize {
        return switch (self) {
            .bool, .i8, .u8 => @alignOf(u8),
            .i16, .u16, .f16 => @alignOf(u16),
            .i32, .u32, .f32, .c64 => @alignOf(f32),
            .i64, .u64, .f64, .c128 => @alignOf(f64),
        };
    }

    /// Returns true if the data type represents a floating-point number.
    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f16, .f32, .f64 => true,
            else => false,
        };
    }

    /// Returns true if the data type represents an integer (signed or unsigned).
    pub fn isInteger(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    /// Returns true if the data type is a signed integer.
    pub fn isSigned(self: DType) bool {
        return switch (self) {
            .i8, .i16, .i32, .i64 => true,
            else => false,
        };
    }

    /// Returns true if the data type is an unsigned integer.
    pub fn isUnsigned(self: DType) bool {
        return switch (self) {
            .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }

    /// Returns true if the data type represents a complex number.
    pub fn isComplex(self: DType) bool {
        return switch (self) {
            .c64, .c128 => true,
            else => false,
        };
    }

    /// Returns true if the data type represents a boolean.
    pub fn isBoolean(self: DType) bool {
        return self == .bool;
    }

    /// Resolves the common promoted data type for binary operations between `a` and `b`.
    pub fn promote(a: DType, b: DType) DType {
        if (a == b) return a;
        if (a == .bool) return b;
        if (b == .bool) return a;

        // If either is complex, promote to complex
        if (a.isComplex() or b.isComplex()) {
            if (a == .c128 or b == .c128 or a == .f64 or b == .f64 or a == .i64 or b == .i64 or a == .u64 or b == .u64) {
                return .c128;
            }
            return .c64;
        }

        // If either is float, the result is floating point.
        if (a.isFloat() or b.isFloat()) {
            if (a.isFloat() and b.isFloat()) {
                return if (a.sizeOf() >= b.sizeOf()) a else b;
            }
            const flt = if (a.isFloat()) a else b;
            const int_t = if (a.isFloat()) b else a;
            // Ensure float has sufficient width to preserve integer range where possible
            if (int_t.sizeOf() >= flt.sizeOf()) {
                return if (int_t.sizeOf() >= 4) .f64 else .f32;
            }
            return flt;
        }

        // Both are integers
        if (a.isSigned() and b.isSigned()) {
            return if (a.sizeOf() >= b.sizeOf()) a else b;
        }
        if (a.isUnsigned() and b.isUnsigned()) {
            return if (a.sizeOf() >= b.sizeOf()) a else b;
        }

        // Mixed signed and unsigned
        const signed_t = if (a.isSigned()) a else b;
        const unsigned_t = if (a.isSigned()) b else a;

        if (unsigned_t.sizeOf() >= signed_t.sizeOf()) {
            return switch (unsigned_t) {
                .u8 => .i16,
                .u16 => .i32,
                .u32 => .i64,
                .u64 => .f64, // Cannot fit u64 and signed in i64, promote to f64
                else => unreachable,
            };
        } else {
            return signed_t;
        }
    }
};

test "dtype size and alignment" {
    try std.testing.expectEqual(@as(usize, 1), DType.bool.sizeOf());
    try std.testing.expectEqual(@as(usize, 1), DType.i8.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.i16.sizeOf());
    try std.testing.expectEqual(@as(usize, 4), DType.i32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.i64.sizeOf());
    try std.testing.expectEqual(@as(usize, 2), DType.f16.sizeOf());
    try std.testing.expectEqual(@as(usize, 4), DType.f32.sizeOf());
    try std.testing.expectEqual(@as(usize, 8), DType.f64.sizeOf());
}

test "dtype introspection" {
    try std.testing.expect(DType.f32.isFloat());
    try std.testing.expect(!DType.i32.isFloat());
    try std.testing.expect(DType.i64.isSigned());
    try std.testing.expect(DType.u64.isUnsigned());
    try std.testing.expect(DType.bool.isBoolean());
}

test "dtype promotion rules" {
    // Identity
    try std.testing.expectEqual(DType.f32, DType.promote(.f32, .f32));
    try std.testing.expectEqual(DType.i32, DType.promote(.i32, .i32));

    // Boolean with type
    try std.testing.expectEqual(DType.i32, DType.promote(.bool, .i32));
    try std.testing.expectEqual(DType.f64, DType.promote(.f64, .bool));

    // Float with float
    try std.testing.expectEqual(DType.f64, DType.promote(.f32, .f64));
    try std.testing.expectEqual(DType.f32, DType.promote(.f16, .f32));

    // Float with int
    try std.testing.expectEqual(DType.f32, DType.promote(.f32, .i8));
    try std.testing.expectEqual(DType.f64, DType.promote(.f32, .i64));
    try std.testing.expectEqual(DType.f64, DType.promote(.f32, .u32));

    // Mixed signed and unsigned
    try std.testing.expectEqual(DType.i16, DType.promote(.i8, .u8));
    try std.testing.expectEqual(DType.i32, DType.promote(.i32, .u8));
    try std.testing.expectEqual(DType.i64, DType.promote(.i32, .u32));
    try std.testing.expectEqual(DType.f64, DType.promote(.i64, .u64));
}
