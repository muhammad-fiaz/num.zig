//! Demonstrates native NZIG v1.0 binary serialization and text CSV I/O in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Create a 2D sample array
    const data = [_]f64{
        1.25, -3.5,  5.75,
        7.0,  -9.25, 11.5,
    };
    var orig = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer orig.deinit();

    // 2. Write to native NZIG v1.0 binary file (.nzig)
    const nzig_path = "example_weights.nzig";
    try num.io.writeFile(allocator, nzig_path, orig);
    defer {
        var io_threaded: std.Io.Threaded = .init(allocator, .{});
        defer io_threaded.deinit();
        const io = io_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, nzig_path) catch {};
    }
    std.debug.print("Successfully serialized array to '{s}'.\n", .{nzig_path});

    // 3. Read back from NZIG file
    var loaded = try num.io.readFile(allocator, nzig_path);
    defer loaded.deinit();

    std.debug.print("Loaded NZIG array:\n  ndim: {d}, shape: [{d}, {d}], dtype: {s}\n", .{
        loaded.ndim,
        loaded.shape_dims[0],
        loaded.shape_dims[1],
        @tagName(loaded.dtype),
    });
    std.debug.print("  val[0, 1] = {d:.2}, val[1, 2] = {d:.2}\n", .{
        try loaded.get(f64, &.{ 0, 1 }),
        try loaded.get(f64, &.{ 1, 2 }),
    });

    // 4. In-memory streaming using MemoryStream
    var buf: [512]u8 = undefined;
    var ms = num.io.MemoryStream.init(&buf);
    try num.io.writeToStream(orig, &ms);

    var rs = num.io.MemoryStream.init(ms.getWritten());
    rs.written = ms.written;
    var stream_loaded = try num.io.readFromStream(allocator, &rs);
    defer stream_loaded.deinit();

    std.debug.print("Memory stream roundtrip successful: {d} elements loaded.\n", .{stream_loaded.elementCount()});
}
