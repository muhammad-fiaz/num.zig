const std = @import("std");
const Allocator = std.mem.Allocator;

/// A generic Graph (Adjacency List).
///
/// Logic: Vertices and Edges.
///
/// Arguments:
///     T: Type of vertex data (must be hashable/equatable).
pub fn Graph(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        adj: std.AutoHashMap(T, std.ArrayListUnmanaged(T)),

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .adj = std.AutoHashMap(T, std.ArrayListUnmanaged(T)).init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            var iter = self.adj.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.deinit(self.allocator);
            }
            self.adj.deinit();
        }

        pub fn addVertex(self: *Self, v: T) !void {
            if (!self.adj.contains(v)) {
                try self.adj.put(v, .{});
            }
        }

        pub fn addEdge(self: *Self, u: T, v: T) !void {
            try self.addVertex(u);
            try self.addVertex(v);
            try self.adj.getPtr(u).?.append(self.allocator, v);
            // For undirected, uncomment:
            // try self.adj.getPtr(v).?.append(self.allocator, u);
        }

        /// Breadth-First Search.
        ///
        /// Logic: Queue-based traversal.
        pub fn bfs(self: Self, start: T) !std.ArrayListUnmanaged(T) {
            var visited = std.AutoHashMap(T, void).init(self.allocator);
            defer visited.deinit();

            var result = std.ArrayListUnmanaged(T){};
            errdefer result.deinit(self.allocator);

            // Simple queue using ArrayListUnmanaged
            var queue = std.ArrayListUnmanaged(T){};
            defer queue.deinit(self.allocator);

            try queue.append(self.allocator, start);
            try visited.put(start, {});

            while (queue.items.len > 0) {
                const current = queue.orderedRemove(0); // Dequeue
                try result.append(self.allocator, current);

                if (self.adj.get(current)) |neighbors| {
                    for (neighbors.items) |neighbor| {
                        if (!visited.contains(neighbor)) {
                            try visited.put(neighbor, {});
                            try queue.append(self.allocator, neighbor);
                        }
                    }
                }
            }

            return result;
        }

        /// Depth-First Search.
        ///
        /// Logic: Stack-based traversal (recursion).
        pub fn dfs(self: Self, start: T) !std.ArrayList(T) {
            var visited = std.AutoHashMap(T, void).init(self.allocator);
            defer visited.deinit();
            var result = std.ArrayList(T).init(self.allocator);
            errdefer result.deinit();

            try self.dfsRecursive(start, &visited, &result);
            return result;
        }

        fn dfsRecursive(self: Self, u: T, visited: *std.AutoHashMap(T, void), result: *std.ArrayList(T)) !void {
            try visited.put(u, {});
            try result.append(u);

            if (self.adj.get(u)) |neighbors| {
                for (neighbors.items) |v| {
                    if (!visited.contains(v)) {
                        try self.dfsRecursive(v, visited, result);
                    }
                }
            }
        }
    };
}
