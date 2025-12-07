const std = @import("std");
const Allocator = std.mem.Allocator;

/// A generic Stack (LIFO).
///
/// Logic: Last In, First Out.
///
/// Arguments:
///     T: Type of data.
pub fn Stack(comptime T: type) type {
    return struct {
        const Self = @This();

        items: std.ArrayListUnmanaged(T),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .items = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
        }

        pub fn push(self: *Self, value: T) !void {
            try self.items.append(self.allocator, value);
        }

        pub fn pop(self: *Self) ?T {
            if (self.items.items.len == 0) return null;
            return self.items.pop();
        }

        pub fn peek(self: Self) ?T {
            if (self.items.items.len == 0) return null;
            return self.items.items[self.items.items.len - 1];
        }

        pub fn isEmpty(self: Self) bool {
            return self.items.items.len == 0;
        }
    };
}
