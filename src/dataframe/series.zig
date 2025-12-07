const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const NDArray = core.NDArray;

pub fn Series(comptime T: type) type {
    return struct {
        const Self = @This();

        name: []const u8,
        data: NDArray(T),
        // Index could be strings, ints, etc. For now, implicit integer index.
        allocator: Allocator,

        pub fn init(allocator: Allocator, name: []const u8, data: NDArray(T)) !Self {
            if (data.rank() != 1) return core.Error.RankMismatch;

            // We take ownership of data, so ensure we clean it up if duplication fails
            var data_mut = data;
            errdefer data_mut.deinit();

            return Self{
                .name = try allocator.dupe(u8, name),
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.name);
            self.data.deinit();
        }

        pub fn print(self: Self) !void {
            std.debug.print("Series: {s}\n", .{self.name});
            // Use NDArray print
            // For now just simple loop
            for (self.data.data, 0..) |val, i| {
                std.debug.print("{d}: {any}\n", .{ i, val });
            }
        }
    };
}
