//! Streaming array input/output interfaces.
//!
//! Provides reader and writer abstractions for streaming array serialization.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const nzig = @import("nzig.zig");

pub const writeToStream = nzig.writeToStream;
pub const readFromStream = nzig.readFromStream;

/// In-memory stream supporting reading, writing, and printing.
pub const MemoryStream = struct {
    buffer: []u8,
    pos: usize = 0,
    written: usize = 0,

    pub fn init(buf: []u8) MemoryStream {
        return .{ .buffer = buf };
    }

    pub fn reset(self: *MemoryStream) void {
        self.pos = 0;
        self.written = 0;
    }

    pub fn getWritten(self: *const MemoryStream) []u8 {
        return self.buffer[0..self.written];
    }

    pub fn writeAll(self: *MemoryStream, bytes: []const u8) error{NoSpaceLeft}!void {
        if (self.written + bytes.len > self.buffer.len) return error.NoSpaceLeft;
        @memcpy(self.buffer[self.written..][0..bytes.len], bytes);
        self.written += bytes.len;
    }

    pub fn writeByte(self: *MemoryStream, b: u8) error{NoSpaceLeft}!void {
        try self.writeAll(&.{b});
    }

    pub fn print(self: *MemoryStream, comptime format_str: []const u8, args: anytype) error{NoSpaceLeft}!void {
        const remaining = self.buffer[self.written..];
        const res = std.fmt.bufPrint(remaining, format_str, args) catch return error.NoSpaceLeft;
        self.written += res.len;
    }

    pub fn readAll(self: *MemoryStream, dest: []u8) error{}!usize {
        const avail = self.written - self.pos;
        const to_read = @min(avail, dest.len);
        @memcpy(dest[0..to_read], self.buffer[self.pos..][0..to_read]);
        self.pos += to_read;
        return to_read;
    }

    pub fn read(self: *MemoryStream, dest: []u8) error{}!usize {
        return self.readAll(dest);
    }

    pub fn readByte(self: *MemoryStream) error{EndOfStream}!u8 {
        if (self.pos >= self.written) return error.EndOfStream;
        const b = self.buffer[self.pos];
        self.pos += 1;
        return b;
    }

    pub fn writer(self: *MemoryStream) *MemoryStream {
        return self;
    }

    pub fn reader(self: *MemoryStream) *MemoryStream {
        return self;
    }
};
