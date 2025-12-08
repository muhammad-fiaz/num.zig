# Statistics

The `stats` module provides comprehensive statistical functions for data analysis, from basic descriptive statistics to advanced operations along specific axes.

## Basic Statistics

### Sum

Calculate the sum of all elements:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var arr = try num.NDArray(f64).arange(allocator, 1.0, 11.0, 1.0);
    defer arr.deinit(allocator);
    
    try arr.print();
    // Output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    const total = try num.stats.sum(allocator, f64, arr);
    std.debug.print("Sum: {d}\n", .{total});
    // Output: Sum: 55.0
}
```

### Product

Calculate the product of all elements:

```zig
var arr = try num.NDArray(f64).arange(allocator, 1.0, 6.0, 1.0);
defer arr.deinit(allocator);

const product = try num.stats.prod(allocator, f64, arr);
std.debug.print("Product of [1,2,3,4,5]: {d}\n", .{product});
// Output: Product of [1,2,3,4,5]: 120.0 (5 factorial)
```

### Mean (Average)

```zig
var arr = try num.NDArray(f64).arange(allocator, 0.0, 100.0, 10.0);
defer arr.deinit(allocator);

try arr.print();
// Output: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

const avg = try num.stats.mean(allocator, f64, arr);
std.debug.print("Mean: {d}\n", .{avg});
// Output: Mean: 45.0
```

### Median

Find the middle value:

```zig
// Odd number of elements
var arr1 = try num.NDArray(f64).arange(allocator, 1.0, 8.0, 1.0);
defer arr1.deinit(allocator);

const median1 = try num.stats.median(allocator, f64, arr1);
std.debug.print("Median of [1..7]: {d}\n", .{median1});
// Output: Median of [1..7]: 4.0

// Even number of elements
var arr2 = try num.NDArray(f64).arange(allocator, 1.0, 9.0, 1.0);
defer arr2.deinit(allocator);

const median2 = try num.stats.median(allocator, f64, arr2);
std.debug.print("Median of [1..8]: {d}\n", .{median2});
// Output: Median of [1..8]: 4.5
```

## Spread Measures

### Variance

Measure of data spread:

```zig
var arr = try num.NDArray(f64).arange(allocator, 1.0, 11.0, 1.0);
defer arr.deinit(allocator);

const variance = try num.stats.variance(allocator, f64, arr);
std.debug.print("Variance: {d:.2}\n", .{variance});
// Output: Variance: 8.25

// Formula: Var(X) = E[(X - μ)²]
// For [1..10]: Var = 8.25
```

### Standard Deviation

Square root of variance:

```zig
var arr = try num.NDArray(f64).arange(allocator, 1.0, 11.0, 1.0);
defer arr.deinit(allocator);

const std_dev = try num.stats.stdDev(allocator, f64, arr);
std.debug.print("Standard Deviation: {d:.2}\n", .{std_dev});
// Output: Standard Deviation: 2.87

// std_dev = √variance = √8.25 ≈ 2.87
```

## Extrema

### Minimum and Maximum

```zig
var arr = try num.NDArray(f64).full(allocator, &.{10}, 0.0);
defer arr.deinit(allocator);

// Fill with random-looking values
arr.data[0] = 5.3;
arr.data[1] = -2.1;
arr.data[2] = 9.7;
arr.data[3] = 0.5;
arr.data[4] = -7.2;
arr.data[5] = 3.3;
arr.data[6] = 12.1;
arr.data[7] = -0.8;
arr.data[8] = 4.2;
arr.data[9] = 6.9;

const min_val = try num.stats.min(allocator, f64, arr);
const max_val = try num.stats.max(allocator, f64, arr);

std.debug.print("Min: {d:.1}\n", .{min_val});
std.debug.print("Max: {d:.1}\n", .{max_val});
// Output:
// Min: -7.2
// Max: 12.1
```

### Argmin and Argmax

Find the index of min/max:

```zig
var arr = try num.NDArray(f64).full(allocator, &.{5}, 0.0);
defer arr.deinit(allocator);
arr.data[0] = 10.0;
arr.data[1] = 5.0;
arr.data[2] = 15.0;
arr.data[3] = 3.0;
arr.data[4] = 8.0;

try arr.print();
// Output: [10.0, 5.0, 15.0, 3.0, 8.0]

const min_idx = try num.stats.argmin(allocator, f64, arr);
const max_idx = try num.stats.argmax(allocator, f64, arr);

std.debug.print("Index of minimum: {}\n", .{min_idx});
std.debug.print("Index of maximum: {}\n", .{max_idx});
// Output:
// Index of minimum: 3 (value 3.0)
// Index of maximum: 2 (value 15.0)
```

## Axis Operations

Compute statistics along specific axes of multidimensional arrays.

### Sum Along Axis

```zig
// Create 3×4 matrix
var matrix = try num.NDArray(f64).arange(allocator, 0.0, 12.0, 1.0);
defer matrix.deinit(allocator);
try matrix.reshape(&.{3, 4});

try matrix.print();
// Output:
// [[0,  1,  2,  3],
//  [4,  5,  6,  7],
//  [8,  9, 10, 11]]

// Sum along axis 0 (down columns)
var col_sums = try num.stats.sumAxis(allocator, f64, matrix, 0);
defer col_sums.deinit(allocator);
try col_sums.print();
// Output: [12, 15, 18, 21]
// (0+4+8, 1+5+9, 2+6+10, 3+7+11)

// Sum along axis 1 (across rows)
var row_sums = try num.stats.sumAxis(allocator, f64, matrix, 1);
defer row_sums.deinit(allocator);
try row_sums.print();
// Output: [6, 22, 38]
// (0+1+2+3, 4+5+6+7, 8+9+10+11)
```

### Mean Along Axis

```zig
var matrix = try num.NDArray(f64).arange(allocator, 0.0, 12.0, 1.0);
defer matrix.deinit(allocator);
try matrix.reshape(&.{3, 4});

// Mean of each column
var col_means = try num.stats.meanAxis(allocator, f64, matrix, 0);
defer col_means.deinit(allocator);
try col_means.print();
// Output: [4.0, 5.0, 6.0, 7.0]

// Mean of each row
var row_means = try num.stats.meanAxis(allocator, f64, matrix, 1);
defer row_means.deinit(allocator);
try row_means.print();
// Output: [1.5, 5.5, 9.5]
```

## Counting and Binning

### Bincount

Count occurrences of non-negative integers:

```zig
var arr = try num.NDArray(u32).full(allocator, &.{10}, 0);
defer arr.deinit(allocator);

// Data: grades from 0-5
arr.data[0] = 3;
arr.data[1] = 2;
arr.data[2] = 5;
arr.data[3] = 3;
arr.data[4] = 2;
arr.data[5] = 4;
arr.data[6] = 3;
arr.data[7] = 5;
arr.data[8] = 2;
arr.data[9] = 4;

try arr.print();
// Output: [3, 2, 5, 3, 2, 4, 3, 5, 2, 4]

var counts = try num.stats.bincount(allocator, u32, arr);
defer counts.deinit(allocator);
try counts.print();
// Output: [0, 0, 3, 3, 2, 2]
// Index:   0  1  2  3  4  5
// Meaning: zero 0's, zero 1's, three 2's, three 3's, two 4's, two 5's
```

### Unique

Find unique elements:

```zig
var arr = try num.NDArray(i32).full(allocator, &.{10}, 0);
defer arr.deinit(allocator);

arr.data[0] = 5;
arr.data[1] = 2;
arr.data[2] = 5;
arr.data[3] = 8;
arr.data[4] = 2;
arr.data[5] = 5;
arr.data[6] = 3;
arr.data[7] = 8;
arr.data[8] = 2;
arr.data[9] = 3;

try arr.print();
// Output: [5, 2, 5, 8, 2, 5, 3, 8, 2, 3]

var unique = try num.stats.unique(allocator, i32, arr);
defer unique.deinit(allocator);
try unique.print();
// Output: [2, 3, 5, 8]
```

## Practical Example: Data Analysis

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Simulated test scores for 3 students across 4 exams
    var scores = try num.NDArray(f64).zeros(allocator, &.{3, 4});
    defer scores.deinit(allocator);
    
    // Student 1: [85, 90, 88, 92]
    scores.data[0] = 85.0;
    scores.data[1] = 90.0;
    scores.data[2] = 88.0;
    scores.data[3] = 92.0;
    
    // Student 2: [78, 82, 85, 80]
    scores.data[4] = 78.0;
    scores.data[5] = 82.0;
    scores.data[6] = 85.0;
    scores.data[7] = 80.0;
    
    // Student 3: [92, 95, 90, 94]
    scores.data[8] = 92.0;
    scores.data[9] = 95.0;
    scores.data[10] = 90.0;
    scores.data[11] = 94.0;
    
    std.debug.print("Scores (students × exams):\n", .{});
    try scores.print();
    
    // Student averages
    var student_avgs = try num.stats.meanAxis(allocator, f64, scores, 1);
    defer student_avgs.deinit(allocator);
    
    std.debug.print("\nStudent Averages:\n", .{});
    for (student_avgs.data, 0..) |avg, i| {
        std.debug.print("  Student {}: {d:.1}\n", .{i + 1, avg});
    }
    
    // Exam averages
    var exam_avgs = try num.stats.meanAxis(allocator, f64, scores, 0);
    defer exam_avgs.deinit(allocator);
    
    std.debug.print("\nExam Averages:\n", .{});
    for (exam_avgs.data, 0..) |avg, i| {
        std.debug.print("  Exam {}: {d:.1}\n", .{i + 1, avg});
    }
    
    // Overall statistics
    const overall_mean = try num.stats.mean(allocator, f64, scores);
    const overall_std = try num.stats.stdDev(allocator, f64, scores);
    const min_score = try num.stats.min(allocator, f64, scores);
    const max_score = try num.stats.max(allocator, f64, scores);
    
    std.debug.print("\nOverall Statistics:\n", .{});
    std.debug.print("  Mean: {d:.1}\n", .{overall_mean});
    std.debug.print("  Std Dev: {d:.2}\n", .{overall_std});
    std.debug.print("  Range: {d:.0} - {d:.0}\n", .{min_score, max_score});
}
```

**Output:**
```
Scores (students × exams):
[[85.0, 90.0, 88.0, 92.0],
 [78.0, 82.0, 85.0, 80.0],
 [92.0, 95.0, 90.0, 94.0]]

Student Averages:
  Student 1: 88.8
  Student 2: 81.2
  Student 3: 92.8

Exam Averages:
  Exam 1: 85.0
  Exam 2: 89.0
  Exam 3: 87.7
  Exam 4: 88.7

Overall Statistics:
  Mean: 87.6
  Std Dev: 5.23
  Range: 78 - 95
```

## Summary Statistics Function

Create a comprehensive statistics summary:

```zig
pub fn describData(allocator: std.mem.Allocator, data: num.NDArray(f64)) !void {
    const mean = try num.stats.mean(allocator, f64, data);
    const median = try num.stats.median(allocator, f64, data);
    const std_dev = try num.stats.stdDev(allocator, f64, data);
    const min_val = try num.stats.min(allocator, f64, data);
    const max_val = try num.stats.max(allocator, f64, data);
    const sum_val = try num.stats.sum(allocator, f64, data);
    
    std.debug.print("Summary Statistics:\n", .{});
    std.debug.print("  Count: {}\n", .{data.size()});
    std.debug.print("  Sum: {d:.2}\n", .{sum_val});
    std.debug.print("  Mean: {d:.2}\n", .{mean});
    std.debug.print("  Median: {d:.2}\n", .{median});
    std.debug.print("  Std Dev: {d:.2}\n", .{std_dev});
    std.debug.print("  Min: {d:.2}\n", .{min_val});
    std.debug.print("  Max: {d:.2}\n", .{max_val});
}
```