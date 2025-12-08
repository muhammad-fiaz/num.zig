const std = @import("std");
const Allocator = std.mem.Allocator;

/// A generic Queue (FIFO).
///
/// Logic: First In, First Out.
///
/// Arguments:
///     T: Type of data.
pub fn Queue(comptime T: type) type {
    return struct {
        const Self = @This();

        // Using a linked list for O(1) enqueue/dequeue
        const Node = struct {
            data: T,
            next: ?*Node,
        };

        head: ?*Node,
        tail: ?*Node,

        pub fn init() Self {
            return .{
                .head = null,
                .tail = null,
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                allocator.destroy(node);
                current = next;
            }
            self.head = null;
            self.tail = null;
        }

        pub fn enqueue(self: *Self, allocator: Allocator, value: T) !void {
            const new_node = try allocator.create(Node);
            new_node.* = .{ .data = value, .next = null };

            if (self.tail) |tail| {
                tail.next = new_node;
            } else {
                self.head = new_node;
            }
            self.tail = new_node;
        }

        pub fn dequeue(self: *Self, allocator: Allocator) ?T {
            if (self.head) |head| {
                const value = head.data;
                self.head = head.next;
                if (self.head == null) {
                    self.tail = null;
                }
                allocator.destroy(head);
                return value;
            }
            return null;
        }

        pub fn isEmpty(self: Self) bool {
            return self.head == null;
        }
    };
}

test "queue" {
    const allocator = std.testing.allocator;
    var q = Queue(i32).init();
    defer q.deinit(allocator);

    try std.testing.expect(q.isEmpty());

    try q.enqueue(allocator, 10);
    try q.enqueue(allocator, 20);

    try std.testing.expect(!q.isEmpty());
    try std.testing.expectEqual(q.dequeue(allocator), 10);
    try std.testing.expectEqual(q.dequeue(allocator), 20);
    try std.testing.expectEqual(q.dequeue(allocator), null);
}
