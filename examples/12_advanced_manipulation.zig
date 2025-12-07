const std = @import("std");
const num = @import("num");
const print = std.debug.print;

pub fn main() !void {
    mainImpl() catch |err| {
        print("Runtime Error: {}\n", .{err});
        print("If you think this is a bug, please report it at: {s}\n", .{num.report_issue_url});
        return err;
    };
}

fn mainImpl() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Example 12: Advanced Array Manipulation\n", .{});

    // 1. Find
    var a = try num.core.NDArray(f32).init(allocator, &.{5});
    @memcpy(a.data, &[_]f32{ 1.0, -2.0, 3.0, -4.0, 5.0 });
    defer a.deinit();

    print("Original array:\n", .{});
    try a.print();

    const isNegative = struct {
        fn call(val: f32) bool {
            return val < 0;
        }
    }.call;

    var indices = try num.manipulation.find(allocator, f32, &a, isNegative);
    defer indices.deinit();
    print("Indices of negative elements:\n", .{});
    try indices.print();

    // 2. Replace Where
    var replaced = try num.manipulation.replaceWhere(allocator, f32, &a, isNegative, 0.0);
    defer replaced.deinit();
    print("Replaced negative elements with 0:\n", .{});
    try replaced.print();

    // 3. Replace Value
    var replaced_val = try num.manipulation.replace(allocator, f32, &a, 3.0, 99.0);
    defer replaced_val.deinit();
    print("Replaced 3.0 with 99.0:\n", .{});
    try replaced_val.print();

    // 4. Delete
    var deleted = try num.manipulation.delete(allocator, f32, &a, &.{ 1, 3 }); // Indices of -2 and -4
    defer deleted.deinit();
    print("Deleted elements at indices 1 and 3:\n", .{});
    try deleted.print();
}
