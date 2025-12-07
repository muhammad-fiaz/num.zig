const std = @import("std");
const num = @import("num");

fn printArray(comptime T: type, arr: num.NDArray(T)) !void {
    if (arr.rank() == 1) {
        std.debug.print("[ ", .{});
        for (0..arr.shape[0]) |i| {
            std.debug.print("{d:.2} ", .{try arr.get(&.{i})});
        }
        std.debug.print("]\n", .{});
    } else if (arr.rank() == 2) {
        std.debug.print("[\n", .{});
        for (0..arr.shape[0]) |i| {
            std.debug.print("  [ ", .{});
            for (0..arr.shape[1]) |j| {
                std.debug.print("{d:.2} ", .{try arr.get(&.{i, j})});
            }
            std.debug.print("]\n", .{});
        }
        std.debug.print("]\n", .{});
    } else {
        std.debug.print("NDArray(rank={d})\n", .{arr.rank()});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Testing New Features...\n", .{});

    // --- Test CSV I/O ---
    std.debug.print("\n--- CSV I/O ---\n", .{});
    
    // Create a dummy CSV file
    const csv_content = 
        \\1.0,2.0,3.0
        \\4.0,5.0,6.0
        \\7.0,8.0,9.0
    ;
    const cwd = std.fs.cwd();
    try cwd.writeFile(.{ .sub_path = "test.csv", .data = csv_content });
    defer cwd.deleteFile("test.csv") catch {};

    // Read CSV
    var csv_arr = try num.io.readCSV(allocator, f64, "test.csv");
    defer csv_arr.deinit();
    
    std.debug.print("Read CSV:\n", .{});
    try printArray(f64, csv_arr);

    // Write CSV
    try num.io.writeCSV(f64, csv_arr, "test_out.csv");
    defer cwd.deleteFile("test_out.csv") catch {};
    std.debug.print("Wrote CSV to test_out.csv\n", .{});

    // --- Test LU Decomposition ---
    std.debug.print("\n--- LU Decomposition ---\n", .{});
    
    var mat = try num.NDArray(f64).init(allocator, &.{3, 3});
    defer mat.deinit();
    
    // A = [[4, 3], [6, 3]] -> L=[[1,0],[1.5,1]], U=[[4,3],[0,-1.5]]
    // Let use a 3x3 example:
    // 1 2 3
    // 4 5 6
    // 7 8 10 (changed 9 to 10 to be non-singular)
    mat.data[0] = 1; mat.data[1] = 2; mat.data[2] = 3;
    mat.data[3] = 4; mat.data[4] = 5; mat.data[5] = 6;
    mat.data[6] = 7; mat.data[7] = 8; mat.data[8] = 10;

    std.debug.print("Matrix A:\n", .{});
    try printArray(f64, mat);

    var lu_res = try num.linalg.lu(f64, allocator, &mat);
    defer {
        lu_res.l.deinit();
        lu_res.u.deinit();
    }

    std.debug.print("L:\n", .{});
    try printArray(f64, lu_res.l);
    std.debug.print("U:\n", .{});
    try printArray(f64, lu_res.u);

    // --- Test Boolean Masking ---
    std.debug.print("\n--- Boolean Masking ---\n", .{});
    var data = try num.NDArray(f64).init(allocator, &.{5});
    defer data.deinit();
    data.data[0] = 10; data.data[1] = 20; data.data[2] = 30; data.data[3] = 40; data.data[4] = 50;

    var mask = try num.NDArray(bool).init(allocator, &.{5});
    defer mask.deinit();
    // Mask: [true, false, true, false, true]
    mask.data[0] = true; mask.data[1] = false; mask.data[2] = true; mask.data[3] = false; mask.data[4] = true;

    var masked = try num.indexing.booleanMask(allocator, f64, data, mask);
    defer masked.deinit();

    std.debug.print("Data: ", .{});
    try printArray(f64, data);
    std.debug.print("Mask: ", .{});
    // printArray(bool, mask) won't work because printArray expects {d:.2} format which is for numbers.
    // I'll skip printing mask or implement printArrayBool
    std.debug.print("Mask: [true, false, true, false, true]\n", .{});
    
    std.debug.print("Masked Result (should be 10, 30, 50):\n", .{});
    try printArray(f64, masked);

}
