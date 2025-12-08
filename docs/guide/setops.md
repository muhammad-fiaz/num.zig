# Set Operations

The `setops` module provides mathematical set operations for 1D arrays, treating them as sets of unique elements. Operations include finding unique elements, intersections, unions, differences, and membership testing.

## Unique Elements

Find unique elements in an array:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Array with duplicates
    const shape = [_]usize{10};
    const data = [_]i32{ 1, 2, 3, 2, 4, 1, 5, 3, 6, 2 };
    var array = num.core.NDArray(i32).init(&shape, @constCast(&data));
    
    var unique_vals = try num.setops.unique(allocator, i32, array);
    defer unique_vals.deinit(allocator);
    
    std.debug.print("Original: {any}\n", .{data});
    std.debug.print("Unique: {any}\n", .{unique_vals.data});
}
// Output:
// Original: [1, 2, 3, 2, 4, 1, 5, 3, 6, 2]
// Unique: [1, 2, 3, 4, 5, 6]
```

## Element Membership (in1d)

Test whether elements of one array are in another:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test array
    const shape1 = [_]usize{5};
    const data1 = [_]i32{ 1, 3, 5, 7, 9 };
    var ar1 = num.core.NDArray(i32).init(&shape1, @constCast(&data1));
    
    // Reference set
    const shape2 = [_]usize{7};
    const data2 = [_]i32{ 2, 3, 5, 7, 11, 13, 17 };
    var ar2 = num.core.NDArray(i32).init(&shape2, @constCast(&data2));
    
    var membership = try num.setops.in1d(allocator, i32, ar1, ar2);
    defer membership.deinit(allocator);
    
    std.debug.print("Test values: {any}\n", .{data1});
    std.debug.print("In set: {any}\n", .{membership.data});
}
// Output:
// Test values: [1, 3, 5, 7, 9]
// In set: [false, true, true, true, false]
// (3, 5, 7 are in the reference set)
```

## Intersection (intersect1d)

Find common elements between two arrays:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Set A: prime numbers under 20
    const shape_a = [_]usize{8};
    const data_a = [_]i32{ 2, 3, 5, 7, 11, 13, 17, 19 };
    var set_a = num.core.NDArray(i32).init(&shape_a, @constCast(&data_a));
    
    // Set B: Fibonacci numbers under 20
    const shape_b = [_]usize{7};
    const data_b = [_]i32{ 1, 1, 2, 3, 5, 8, 13 };
    var set_b = num.core.NDArray(i32).init(&shape_b, @constCast(&data_b));
    
    var intersection = try num.setops.intersect1d(allocator, i32, set_a, set_b);
    defer intersection.deinit(allocator);
    
    std.debug.print("Primes: {any}\n", .{data_a});
    std.debug.print("Fibonacci: {any}\n", .{data_b});
    std.debug.print("Intersection: {any}\n", .{intersection.data});
}
// Output:
// Primes: [2, 3, 5, 7, 11, 13, 17, 19]
// Fibonacci: [1, 1, 2, 3, 5, 8, 13]
// Intersection: [2, 3, 5, 13]
```

## Union (union1d)

Find all unique elements from both arrays:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape_a = [_]usize{5};
    const data_a = [_]i32{ 1, 2, 3, 4, 5 };
    var set_a = num.core.NDArray(i32).init(&shape_a, @constCast(&data_a));
    
    const shape_b = [_]usize{5};
    const data_b = [_]i32{ 4, 5, 6, 7, 8 };
    var set_b = num.core.NDArray(i32).init(&shape_b, @constCast(&data_b));
    
    var union_set = try num.setops.union1d(allocator, i32, set_a, set_b);
    defer union_set.deinit(allocator);
    
    std.debug.print("Set A: {any}\n", .{data_a});
    std.debug.print("Set B: {any}\n", .{data_b});
    std.debug.print("Union: {any}\n", .{union_set.data});
}
// Output:
// Set A: [1, 2, 3, 4, 5]
// Set B: [4, 5, 6, 7, 8]
// Union: [1, 2, 3, 4, 5, 6, 7, 8]
```

## Set Difference (setdiff1d)

Find elements in first array that are not in second:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // All students
    const shape_all = [_]usize{8};
    const data_all = [_]i32{ 101, 102, 103, 104, 105, 106, 107, 108 };
    var all_students = num.core.NDArray(i32).init(&shape_all, @constCast(&data_all));
    
    // Students who passed
    const shape_passed = [_]usize{5};
    const data_passed = [_]i32{ 102, 104, 105, 107, 108 };
    var passed = num.core.NDArray(i32).init(&shape_passed, @constCast(&data_passed));
    
    var failed = try num.setops.setdiff1d(allocator, i32, all_students, passed);
    defer failed.deinit(allocator);
    
    std.debug.print("All students: {any}\n", .{data_all});
    std.debug.print("Passed: {any}\n", .{data_passed});
    std.debug.print("Failed: {any}\n", .{failed.data});
}
// Output:
// All students: [101, 102, 103, 104, 105, 106, 107, 108]
// Passed: [102, 104, 105, 107, 108]
// Failed: [101, 103, 106]
```

## Symmetric Difference (setxor1d)

Find elements in either array but not in both:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Group A preferences
    const shape_a = [_]usize{6};
    const data_a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var group_a = num.core.NDArray(i32).init(&shape_a, @constCast(&data_a));
    
    // Group B preferences
    const shape_b = [_]usize{6};
    const data_b = [_]i32{ 4, 5, 6, 7, 8, 9 };
    var group_b = num.core.NDArray(i32).init(&shape_b, @constCast(&data_b));
    
    var exclusive = try num.setops.setxor1d(allocator, i32, group_a, group_b);
    defer exclusive.deinit(allocator);
    
    std.debug.print("Group A: {any}\n", .{data_a});
    std.debug.print("Group B: {any}\n", .{data_b});
    std.debug.print("Exclusive choices: {any}\n", .{exclusive.data});
}
// Output:
// Group A: [1, 2, 3, 4, 5, 6]
// Group B: [4, 5, 6, 7, 8, 9]
// Exclusive choices: [1, 2, 3, 7, 8, 9]
// (Elements unique to each group)
```

## Practical Example: Data Validation

Validate that all required IDs are present:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Required product IDs
    const shape_req = [_]usize{5};
    const data_req = [_]i32{ 1001, 1002, 1003, 1004, 1005 };
    var required = num.core.NDArray(i32).init(&shape_req, @constCast(&data_req));
    
    // Received product IDs
    const shape_recv = [_]usize{6};
    const data_recv = [_]i32{ 1001, 1002, 1003, 1006, 1007, 1008 };
    var received = num.core.NDArray(i32).init(&shape_recv, @constCast(&data_recv));
    
    // Find missing items
    var missing = try num.setops.setdiff1d(allocator, i32, required, received);
    defer missing.deinit(allocator);
    
    // Find extra items
    var extra = try num.setops.setdiff1d(allocator, i32, received, required);
    defer extra.deinit(allocator);
    
    std.debug.print("Validation Report:\n", .{});
    std.debug.print("  Required: {any}\n", .{data_req});
    std.debug.print("  Received: {any}\n", .{data_recv});
    std.debug.print("  Missing: {any}\n", .{missing.data});
    std.debug.print("  Extra: {any}\n", .{extra.data});
}
// Output:
// Validation Report:
//   Required: [1001, 1002, 1003, 1004, 1005]
//   Received: [1001, 1002, 1003, 1006, 1007, 1008]
//   Missing: [1004, 1005]
//   Extra: [1006, 1007, 1008]
```

## Practical Example: Category Analysis

Analyze overlap between different categories:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Customer segments
    const shape_premium = [_]usize{6};
    const data_premium = [_]i32{ 101, 102, 103, 104, 105, 106 };
    var premium_customers = num.core.NDArray(i32).init(&shape_premium, @constCast(&data_premium));
    
    const shape_active = [_]usize{8};
    const data_active = [_]i32{ 102, 104, 105, 107, 108, 109, 110, 111 };
    var active_customers = num.core.NDArray(i32).init(&shape_active, @constCast(&data_active));
    
    // Premium AND Active
    var premium_active = try num.setops.intersect1d(allocator, i32, premium_customers, active_customers);
    defer premium_active.deinit(allocator);
    
    // Premium OR Active
    var any_segment = try num.setops.union1d(allocator, i32, premium_customers, active_customers);
    defer any_segment.deinit(allocator);
    
    // Premium but NOT Active
    var premium_only = try num.setops.setdiff1d(allocator, i32, premium_customers, active_customers);
    defer premium_only.deinit(allocator);
    
    std.debug.print("Customer Segmentation:\n", .{});
    std.debug.print("  Premium & Active: {d} customers\n", .{premium_active.shape[0]});
    std.debug.print("  Total in segments: {d} customers\n", .{any_segment.shape[0]});
    std.debug.print("  Premium only: {any}\n", .{premium_only.data});
}
// Output:
// Customer Segmentation:
//   Premium & Active: 3 customers
//   Total in segments: 11 customers
//   Premium only: [101, 103, 106]
```