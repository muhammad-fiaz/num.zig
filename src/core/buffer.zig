//! 64-byte aligned raw memory buffer backend.
//!
//! Provides aligned memory allocation for SIMD vector throughput, element slicing,
//! and lifecycle management.

const std = @import("std");
const DType = @import("dtype.zig").DType;
const DTypeError = @import("error.zig").DTypeError;

/// Required byte alignment for SIMD vector operations (AVX-512, AVX2, NEON).
pub const SIMD_ALIGNMENT: usize = 64;

/// Managed 64-byte aligned storage buffer.
pub const Buffer = struct {
    allocator: std.mem.Allocator,
    bytes: []align(SIMD_ALIGNMENT) u8,
    dtype: DType,
    len: usize,

    /// Allocates uninitialized memory for `len` elements of type `dtype`.
    pub fn alloc(allocator: std.mem.Allocator, dtype: DType, len: usize) std.mem.Allocator.Error!Buffer {
        const byte_len = std.math.mul(usize, len, dtype.sizeOf()) catch return std.mem.Allocator.Error.OutOfMemory;
        const bytes = try allocator.alignedAlloc(u8, .fromByteUnits(SIMD_ALIGNMENT), byte_len);
        return .{
            .allocator = allocator,
            .bytes = bytes,
            .dtype = dtype,
            .len = len,
        };
    }

    /// Allocates zero-initialized memory for `len` elements of type `dtype`.
    pub fn allocZeroed(allocator: std.mem.Allocator, dtype: DType, len: usize) std.mem.Allocator.Error!Buffer {
        const buf = try alloc(allocator, dtype, len);
        @memset(buf.bytes, 0);
        return buf;
    }

    /// Allocates a new buffer initialized with copies of elements from a slice.
    pub fn fromSlice(allocator: std.mem.Allocator, comptime T: type, data: []const T) std.mem.Allocator.Error!Buffer {
        const dtype = DType.fromType(T);
        var buf = try alloc(allocator, dtype, data.len);
        const typed_slice = buf.asSlice(T);
        @memcpy(typed_slice, data);
        return buf;
    }

    /// Duplicates the buffer and its contents using its own allocator.
    pub fn clone(self: Buffer) std.mem.Allocator.Error!Buffer {
        const new_buf = try alloc(self.allocator, self.dtype, self.len);
        @memcpy(new_buf.bytes, self.bytes);
        return new_buf;
    }

    /// Releases the allocated memory.
    pub fn deinit(self: *Buffer) void {
        self.allocator.free(self.bytes);
        self.bytes = &.{};
        self.len = 0;
    }

    /// Returns a typed mutable slice over the buffer memory.
    pub fn asSlice(self: Buffer, comptime T: type) []T {
        std.debug.assert(DType.fromType(T) == self.dtype);
        const ptr: [*]T = @ptrCast(@alignCast(self.bytes.ptr));
        return ptr[0..self.len];
    }

    /// Returns a typed immutable slice over the buffer memory.
    pub fn asConstSlice(self: Buffer, comptime T: type) []const T {
        std.debug.assert(DType.fromType(T) == self.dtype);
        const ptr: [*]const T = @ptrCast(@alignCast(self.bytes.ptr));
        return ptr[0..self.len];
    }

    /// Reads a scalar element of type `T` at the specified index.
    pub fn get(self: Buffer, comptime T: type, index: usize) T {
        std.debug.assert(index < self.len);
        return self.asConstSlice(T)[index];
    }

    /// Writes a scalar element of type `T` at the specified index.
    pub fn set(self: Buffer, comptime T: type, index: usize, value: T) void {
        std.debug.assert(index < self.len);
        self.asSlice(T)[index] = value;
    }

    /// Returns the raw byte slice.
    pub fn rawBytes(self: Buffer) []u8 {
        return self.bytes;
    }

    /// Returns the raw const byte slice.
    pub fn rawBytesConst(self: Buffer) []const u8 {
        return self.bytes;
    }
};

test "buffer allocation and typed access" {
    const allocator = std.testing.allocator;

    var buf = try Buffer.allocZeroed(allocator, .f32, 16);
    defer buf.deinit();

    try std.testing.expectEqual(@as(usize, 16), buf.len);
    try std.testing.expectEqual(@as(usize, 64), buf.bytes.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(buf.bytes.ptr) % SIMD_ALIGNMENT);

    const f32_slice = buf.asSlice(f32);
    for (f32_slice, 0..) |*val, i| {
        val.* = @floatFromInt(i * 10);
    }

    try std.testing.expectEqual(@as(f32, 50.0), buf.get(f32, 5));
    buf.set(f32, 5, 99.0);
    try std.testing.expectEqual(@as(f32, 99.0), buf.get(f32, 5));
}

test "buffer clone and fromSlice" {
    const allocator = std.testing.allocator;

    const data = [_]i32{ 1, 2, 3, 4, 5 };
    var buf1 = try Buffer.fromSlice(allocator, i32, &data);
    defer buf1.deinit();

    var buf2 = try buf1.clone();
    defer buf2.deinit();

    try std.testing.expectEqualSlices(i32, &data, buf2.asSlice(i32));
    buf2.set(i32, 0, 100);
    try std.testing.expectEqual(@as(i32, 1), buf1.get(i32, 0));
    try std.testing.expectEqual(@as(i32, 100), buf2.get(i32, 0));
}
