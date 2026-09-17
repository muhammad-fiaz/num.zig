//! Demonstrates covariance, Pearson correlation coefficients, and histograms.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Two correlated variables: x and y = 2x + noise
    const x_vals = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y_vals = [_]f64{ 2.1, 3.9, 6.2, 8.0, 9.8 };

    var x = try num.fromSlice(allocator, f64, .{ .data = &x_vals, .shape = &.{5} });
    defer x.deinit();
    var y = try num.fromSlice(allocator, f64, .{ .data = &y_vals, .shape = &.{5} });
    defer y.deinit();

    // 1. Covariance matrix (2x2)
    var cov = try num.stats.covariance(x, y, .{});
    defer cov.deinit();

    std.debug.print("Covariance Matrix (2x2):\n", .{});
    std.debug.print("  [{d:.4}, {d:.4}]\n", .{ try cov.get(f64, &.{ 0, 0 }), try cov.get(f64, &.{ 0, 1 }) });
    std.debug.print("  [{d:.4}, {d:.4}]\n", .{ try cov.get(f64, &.{ 1, 0 }), try cov.get(f64, &.{ 1, 1 }) });

    // 2. Correlation coefficient matrix (2x2)
    var corr = try num.stats.corrcoef(x, y, .{});
    defer corr.deinit();

    std.debug.print("Pearson Correlation Matrix (2x2):\n", .{});
    std.debug.print("  r(x, y): {d:.4}\n", .{try corr.get(f64, &.{ 0, 1 })});

    // 3. Histogram computation into 3 bins
    var hist = try num.stats.histogram(x, .{ .bins = 3 });
    defer hist.deinit();

    std.debug.print("Histogram counts over 3 bins:\n  [", .{});
    for (0..hist.counts.elementCount()) |i| {
        std.debug.print("{d} ", .{try hist.counts.get(i64, &.{i})});
    }
    std.debug.print("]\n", .{});
}
