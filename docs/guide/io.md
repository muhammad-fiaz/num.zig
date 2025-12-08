# Input / Output

The `io` module provides functions for saving and loading NDArrays to and from disk in both binary and CSV formats. Essential for data persistence, sharing datasets, and integration with other tools.

## Saving Arrays (Binary Format)

Save an NDArray to a binary file:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create array to save
    const shape = [_]usize{ 3, 4 };
    const data = [_]f32{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
    };
    var array = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // Save to binary file
    try num.io.save(allocator, f32, array, "matrix_data.bin");
    
    std.debug.print("Array saved to matrix_data.bin\n", .{});
}
// Output:
// Array saved to matrix_data.bin
// File contains shape metadata and raw binary data
```

## Loading Arrays (Binary Format)

Load an NDArray from a binary file:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load from binary file
    var loaded = try num.io.load(allocator, f32, "matrix_data.bin");
    defer loaded.deinit(allocator);
    
    std.debug.print("Loaded array shape: {any}\n", .{loaded.shape});
    std.debug.print("First 4 values: {any}\n", .{loaded.data[0..4]});
}
// Output:
// Loaded array shape: [3, 4]
// First 4 values: [1.0, 2.0, 3.0, 4.0]
```

## Writing CSV Files

Export an NDArray to CSV format for use in spreadsheets or other tools:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create 2D array (matrix)
    const shape = [_]usize{ 3, 3 };
    const data = [_]f32{
        1.5, 2.7, 3.2,
        4.1, 5.9, 6.3,
        7.0, 8.8, 9.6,
    };
    var matrix = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // Write to CSV
    try num.io.writeCSV(f32, matrix, "output_matrix.csv");
    
    std.debug.print("Matrix written to output_matrix.csv\n", .{});
}
// Output:
// Matrix written to output_matrix.csv
// File contents:
// 1.5,2.7,3.2
// 4.1,5.9,6.3
// 7.0,8.8,9.6
```

## Reading CSV Files

Import data from CSV files:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Read from CSV file
    var loaded = try num.io.readCSV(allocator, f32, "output_matrix.csv");
    defer loaded.deinit(allocator);
    
    std.debug.print("Loaded CSV shape: {any}\n", .{loaded.shape});
    std.debug.print("Data:\n", .{});
    for (0..3) |i| {
        for (0..3) |j| {
            std.debug.print("{d:.1} ", .{loaded.data[i * 3 + j]});
        }
        std.debug.print("\n", .{});
    }
}
// Output:
// Loaded CSV shape: [3, 3]
// Data:
// 1.5 2.7 3.2
// 4.1 5.9 6.3
// 7.0 8.8 9.6
```

## Writing Array to Stream

Write an NDArray to any writer interface:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{5};
    const data = [_]i32{ 10, 20, 30, 40, 50 };
    var array = num.core.NDArray(i32).init(&shape, @constCast(&data));
    
    // Write to stdout
    const stdout = std.io.getStdOut().writer();
    try num.io.writeArray(allocator, i32, array, stdout);
    
    std.debug.print("\nArray written to stream\n", .{});
}
// Output:
// [Binary data written to stream]
// Array written to stream
```

## Reading Array from Stream

Read an NDArray from any reader interface:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Open file for reading
    const file = try std.fs.cwd().openFile("matrix_data.bin", .{});
    defer file.close();
    
    const reader = file.reader();
    var array = try num.io.readArray(allocator, f32, reader);
    defer array.deinit(allocator);
    
    std.debug.print("Read array with shape: {any}\n", .{array.shape});
}
// Output:
// Read array with shape: [3, 4]
```

## Practical Example: Dataset Management

Save and load a complete dataset:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create training data (100 samples, 5 features)
    const shape = [_]usize{ 100, 5 };
    var data = try allocator.alloc(f32, 100 * 5);
    defer allocator.free(data);
    
    // Populate with synthetic data
    for (0..500) |i| {
        data[i] = @as(f32, @floatFromInt(i % 10)) * 0.1;
    }
    
    var training_set = num.core.NDArray(f32).init(&shape, data);
    
    // Save dataset
    try num.io.save(allocator, f32, training_set, "training_data.bin");
    std.debug.print("Training data saved\n", .{});
    
    // Later: load dataset
    var loaded_data = try num.io.load(allocator, f32, "training_data.bin");
    defer loaded_data.deinit(allocator);
    
    std.debug.print("Loaded {d} samples with {d} features each\n", .{
        loaded_data.shape[0],
        loaded_data.shape[1],
    });
}
// Output:
// Training data saved
// Loaded 100 samples with 5 features each
```

## Practical Example: Export Results to CSV

Compute results and export for analysis in Excel/Python:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Simulation results: time vs measurements
    const n_steps: usize = 10;
    const shape = [_]usize{ n_steps, 3 };
    var results = try allocator.alloc(f32, n_steps * 3);
    defer allocator.free(results);
    
    // Columns: time, position, velocity
    for (0..n_steps) |i| {
        const t = @as(f32, @floatFromInt(i));
        results[i * 3 + 0] = t;           // time
        results[i * 3 + 1] = t * t * 0.5; // position
        results[i * 3 + 2] = t;           // velocity
    }
    
    var result_table = num.core.NDArray(f32).init(&shape, results);
    
    // Export to CSV
    try num.io.writeCSV(f32, result_table, "simulation_results.csv");
    
    std.debug.print("Results exported to simulation_results.csv\n", .{});
    std.debug.print("Columns: Time, Position, Velocity\n", .{});
}
// Output:
// Results exported to simulation_results.csv
// Columns: Time, Position, Velocity
// File contains:
// 0.0,0.0,0.0
// 1.0,0.5,1.0
// 2.0,2.0,2.0
// ...
```

## Practical Example: Data Pipeline

Load CSV, process, save binary for fast reloading:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Data Processing Pipeline ===\n", .{});
    
    // Step 1: Load raw CSV data
    std.debug.print("1. Loading CSV...\n", .{});
    var raw_data = try num.io.readCSV(allocator, f32, "raw_input.csv");
    defer raw_data.deinit(allocator);
    
    // Step 2: Process data (example: normalize)
    std.debug.print("2. Processing data...\n", .{});
    for (0..raw_data.data.len) |i| {
        raw_data.data[i] = raw_data.data[i] / 100.0;
    }
    
    // Step 3: Save as binary for fast loading
    std.debug.print("3. Saving processed data...\n", .{});
    try num.io.save(allocator, f32, raw_data, "processed_data.bin");
    
    std.debug.print("Pipeline complete!\n", .{});
    std.debug.print("Processed shape: {any}\n", .{raw_data.shape});
}
// Output:
// === Data Processing Pipeline ===
// 1. Loading CSV...
// 2. Processing data...
// 3. Saving processed data...
// Pipeline complete!
// Processed shape: [N, M]
```