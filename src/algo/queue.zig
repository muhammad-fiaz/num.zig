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
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .head = null,
                .tail = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                self.allocator.destroy(node);
                current = next;
            }
            self.head = null;
            self.tail = null;
        }

        pub fn enqueue(self: *Self, value: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .data = value, .next = null };

            if (self.tail) |tail| {
                tail.next = new_node;
            } else {
                self.head = new_node;
            }
            self.tail = new_node;
        }

        pub fn dequeue(self: *Self) ?T {
            if (self.head) |head| {
                const value = head.data;
                self.head = head.next;
                if (self.head == null) {
                    self.tail = null;
                }
                self.allocator.destroy(head);
                return value;
            }
            return null;
        }

        pub fn isEmpty(self: Self) bool {
            return self.head == null;
        }
    };
}
