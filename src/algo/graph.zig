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

        adj: std.AutoHashMapUnmanaged(T, std.ArrayListUnmanaged(T)),

        pub fn init() Self {
            return .{
                .adj = std.AutoHashMapUnmanaged(T, std.ArrayListUnmanaged(T)){},
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            var iter = self.adj.iterator();
            while (iter.next()) |entry| {
                entry.value_ptr.deinit(allocator);
            }
            self.adj.deinit(allocator);
        }

        pub fn addVertex(self: *Self, allocator: Allocator, v: T) !void {
            if (!self.adj.contains(v)) {
                try self.adj.put(allocator, v, .{});
            }
        }

        pub fn addEdge(self: *Self, allocator: Allocator, u: T, v: T) !void {
            try self.addVertex(allocator, u);
            try self.addVertex(allocator, v);
            try self.adj.getPtr(u).?.append(allocator, v);
            // For undirected, uncomment:
            // try self.adj.getPtr(v).?.append(allocator, u);
        }

        /// Breadth-First Search.
        ///
        /// Logic: Queue-based traversal.
        pub fn bfs(self: Self, allocator: Allocator, start: T) !std.ArrayListUnmanaged(T) {
            var visited = std.AutoHashMap(T, void).init(allocator);
            defer visited.deinit();

            var result = std.ArrayListUnmanaged(T){};
            errdefer result.deinit(allocator);

            // Simple queue using ArrayListUnmanaged
            var queue = std.ArrayListUnmanaged(T){};
            defer queue.deinit(allocator);

            try queue.append(allocator, start);
            try visited.put(start, {});

            while (queue.items.len > 0) {
                const current = queue.orderedRemove(0); // Dequeue
                try result.append(allocator, current);

                if (self.adj.get(current)) |neighbors| {
                    for (neighbors.items) |neighbor| {
                        if (!visited.contains(neighbor)) {
                            try visited.put(neighbor, {});
                            try queue.append(allocator, neighbor);
                        }
                    }
                }
            }

            return result;
        }

        /// Depth-First Search.
        ///
        /// Logic: Stack-based traversal (recursion).
        pub fn dfs(self: Self, allocator: Allocator, start: T) !std.ArrayList(T) {
            var visited = std.AutoHashMap(T, void).init(allocator);
            defer visited.deinit();
            var result = std.ArrayList(T).init(allocator);
            errdefer result.deinit();

            try self.dfsRecursive(allocator, start, &visited, &result);
            return result;
        }

        fn dfsRecursive(self: Self, allocator: Allocator, u: T, visited: *std.AutoHashMap(T, void), result: *std.ArrayList(T)) !void {
            try visited.put(u, {});
            try result.append(u);

            if (self.adj.get(u)) |neighbors| {
                for (neighbors.items) |v| {
                    if (!visited.contains(v)) {
                        try self.dfsRecursive(allocator, v, visited, result);
                    }
                }
            }
        }
    };
}
