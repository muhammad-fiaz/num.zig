const std = @import("std");
const Allocator = std.mem.Allocator;

/// A generic singly linked list.
///
/// Logic: Standard linked list implementation.
///
/// Arguments:
///     T: Type of data stored in nodes.
pub fn LinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            next: ?*Node,
        };

        head: ?*Node,

        /// Initialize a new linked list.
        ///
        /// Logic: head = null
        ///
        /// Returns:
        ///     New LinkedList instance.
        pub fn init() Self {
            return .{
                .head = null,
            };
        }

        /// Deinitialize the list and free all nodes.
        ///
        /// Logic: traverse and free
        pub fn deinit(self: *Self, allocator: Allocator) void {
            var current = self.head;
            while (current) |node| {
                const next = node.next;
                allocator.destroy(node);
                current = next;
            }
            self.head = null;
        }

        /// Append a value to the end of the list.
        ///
        /// Logic: traverse to end and link new node
        ///
        /// Arguments:
        ///     allocator: Allocator to use for nodes.
        ///     value: Value to append.
        pub fn append(self: *Self, allocator: Allocator, value: T) !void {
            const new_node = try allocator.create(Node);
            new_node.* = .{ .data = value, .next = null };

            if (self.head == null) {
                self.head = new_node;
            } else {
                var current = self.head.?;
                while (current.next) |next| {
                    current = next;
                }
                current.next = new_node;
            }
        }

        /// Prepend a value to the start of the list.
        ///
        /// Logic: new_node.next = head; head = new_node
        ///
        /// Arguments:
        ///     allocator: Allocator to use for nodes.
        ///     value: Value to prepend.
        pub fn prepend(self: *Self, allocator: Allocator, value: T) !void {
            const new_node = try allocator.create(Node);
            new_node.* = .{ .data = value, .next = self.head };
            self.head = new_node;
        }

        /// Find a value in the list.
        ///
        /// Logic: traverse and compare
        ///
        /// Arguments:
        ///     value: Value to find.
        ///
        /// Returns:
        ///     Pointer to node if found, null otherwise.
        pub fn find(self: Self, value: T) ?*Node {
            var current = self.head;
            while (current) |node| {
                if (node.data == value) return node;
                current = node.next;
            }
            return null;
        }

        /// Delete the first occurrence of a value.
        ///
        /// Logic: traverse, find, unlink, free
        ///
        /// Arguments:
        ///     allocator: Allocator.
        ///     value: Value to delete.
        ///
        /// Returns:
        ///     bool (true if deleted, false if not found).
        pub fn delete(self: *Self, allocator: Allocator, value: T) bool {
            var current = self.head;
            var prev: ?*Node = null;

            while (current) |node| {
                if (node.data == value) {
                    if (prev) |p| {
                        p.next = node.next;
                    } else {
                        self.head = node.next;
                    }
                    allocator.destroy(node);
                    return true;
                }
                prev = node;
                current = node.next;
            }
            return false;
        }
    };
}

/// A generic doubly linked list.
///
/// Logic: Standard doubly linked list.
///
/// Arguments:
///     T: Type of data stored in nodes.
pub fn DoublyLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            next: ?*Node,
            prev: ?*Node,
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

        pub fn append(self: *Self, value: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .data = value, .next = null, .prev = self.tail };

            if (self.tail) |tail| {
                tail.next = new_node;
            } else {
                self.head = new_node;
            }
            self.tail = new_node;
        }

        pub fn prepend(self: *Self, value: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .data = value, .next = self.head, .prev = null };

            if (self.head) |head| {
                head.prev = new_node;
            } else {
                self.tail = new_node;
            }
            self.head = new_node;
        }

        pub fn delete(self: *Self, value: T) bool {
            var current = self.head;
            while (current) |node| {
                if (node.data == value) {
                    if (node.prev) |prev| {
                        prev.next = node.next;
                    } else {
                        self.head = node.next;
                    }

                    if (node.next) |next| {
                        next.prev = node.prev;
                    } else {
                        self.tail = node.prev;
                    }

                    self.allocator.destroy(node);
                    return true;
                }
                current = node.next;
            }
            return false;
        }
    };
}

/// A generic circular linked list.
///
/// Logic: Last node points to head.
///
/// Arguments:
///     T: Type of data stored in nodes.
pub fn CircularLinkedList(comptime T: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            data: T,
            next: ?*Node,
        };

        head: ?*Node,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .head = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.head == null) return;

            var current = self.head;
            while (true) {
                const next = current.?.next;
                self.allocator.destroy(current.?);
                current = next;
                if (current == self.head) break;
            }
            self.head = null;
        }

        pub fn append(self: *Self, value: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .data = value, .next = null };

            if (self.head == null) {
                self.head = new_node;
                new_node.next = new_node;
            } else {
                var current = self.head.?;
                while (current.next != self.head) {
                    current = current.next.?;
                }
                current.next = new_node;
                new_node.next = self.head;
            }
        }
    };
}

test "linked list" {
    const allocator = std.testing.allocator;
    var list = LinkedList(i32).init();
    defer list.deinit(allocator);

    try list.append(allocator, 1);
    try list.append(allocator, 2);
    try list.prepend(allocator, 0);

    try std.testing.expectEqual(list.head.?.data, 0);
    try std.testing.expectEqual(list.head.?.next.?.data, 1);
    try std.testing.expectEqual(list.head.?.next.?.next.?.data, 2);

    try std.testing.expect(list.find(1) != null);
    try std.testing.expect(list.find(99) == null);

    _ = list.delete(allocator, 1);
    try std.testing.expect(list.find(1) == null);
}
