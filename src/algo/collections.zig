const std = @import("std");
const Allocator = std.mem.Allocator;

/// A generic Priority Queue.
///
/// Logic: Binary Heap.
///
/// Arguments:
///     T: Type of data.
///     compare: Comparison function (less than).
pub fn PriorityQueue(comptime T: type, comptime compare: fn (void, T, T) bool) type {
    return std.PriorityQueue(T, void, compare);
}

/// A generic HashSet.
///
/// Logic: Hash Map with void values.
///
/// Arguments:
///     T: Type of data.
pub fn HashSet(comptime T: type) type {
    return struct {
        const Self = @This();
        map: std.AutoHashMap(T, void),

        pub fn init(allocator: Allocator) Self {
            return .{
                .map = std.AutoHashMap(T, void).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.map.deinit();
        }

        pub fn add(self: *Self, value: T) !void {
            try self.map.put(value, {});
        }

        pub fn contains(self: Self, value: T) bool {
            return self.map.contains(value);
        }

        pub fn remove(self: *Self, value: T) bool {
            return self.map.remove(value);
        }

        pub fn count(self: Self) usize {
            return self.map.count();
        }
    };
}

test "collections" {
    const allocator = std.testing.allocator;

    // Test HashSet
    var set = HashSet(i32).init(allocator);
    defer set.deinit();

    try set.add(1);
    try set.add(2);
    try set.add(1); // Duplicate

    try std.testing.expectEqual(set.count(), 2);
    try std.testing.expect(set.contains(1));
    try std.testing.expect(set.contains(2));
    try std.testing.expect(!set.contains(3));

    _ = set.remove(1);
    try std.testing.expect(!set.contains(1));
}
