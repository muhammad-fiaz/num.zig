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

        pub fn init() Self {
            return .{
                .items = .{},
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.items.deinit(allocator);
        }

        pub fn push(self: *Self, allocator: Allocator, value: T) !void {
            try self.items.append(allocator, value);
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

test "stack" {
    const allocator = std.testing.allocator;
    var s = Stack(i32).init();
    defer s.deinit(allocator);

    try std.testing.expect(s.isEmpty());

    try s.push(allocator, 10);
    try s.push(allocator, 20);

    try std.testing.expect(!s.isEmpty());
    try std.testing.expectEqual(s.peek(), 20);
    try std.testing.expectEqual(s.pop(), 20);
    try std.testing.expectEqual(s.pop(), 10);
    try std.testing.expectEqual(s.pop(), null);
}
