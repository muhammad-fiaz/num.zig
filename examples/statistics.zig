//! Demonstrates descriptive statistics: mean, variance, stdDev, median, quantile, percentile.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const data = [_]f64{ 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0, 35.0 };
    var arr = try num.fromSlice(allocator, f64, .{ .data = &data, .shape = &.{8} });
    defer arr.deinit();

    // 1. Mean and Median
    var m = try num.stats.mean(arr, .{});
    defer m.deinit();
    var med = try num.stats.median(arr, .{});
    defer med.deinit();

    // 2. Variance and Standard Deviation
    var v = try num.stats.variance(arr, .{});
    defer v.deinit();
    var s = try num.stats.stdDev(arr, .{});
    defer s.deinit();

    // 3. Percentiles (25th, 50th, 75th)
    var p25 = try num.stats.percentile(arr, 25.0, .{});
    defer p25.deinit();
    var p50 = try num.stats.percentile(arr, 50.0, .{});
    defer p50.deinit();
    var p75 = try num.stats.percentile(arr, 75.0, .{});
    defer p75.deinit();

    std.debug.print("Descriptive Statistics:\n", .{});
    std.debug.print("  Mean:     {d:.2}\n", .{try m.get(f64, &.{})});
    std.debug.print("  Median:   {d:.2}\n", .{try med.get(f64, &.{})});
    std.debug.print("  Variance: {d:.2}\n", .{try v.get(f64, &.{})});
    std.debug.print("  StdDev:   {d:.2}\n", .{try s.get(f64, &.{})});
    std.debug.print("  25th Pct: {d:.2}\n", .{try p25.get(f64, &.{})});
    std.debug.print("  50th Pct: {d:.2}\n", .{try p50.get(f64, &.{})});
    std.debug.print("  75th Pct: {d:.2}\n", .{try p75.get(f64, &.{})});

    // 4. Min / max / range and nearest-rank quantile method
    var lo = try num.stats.min(arr, .{});
    defer lo.deinit();
    var hi = try num.stats.max(arr, .{});
    defer hi.deinit();
    var r = try num.stats.range(arr, .{});
    defer r.deinit();
    var qn = try num.stats.quantileWithOptions(arr, 0.5, .{ .method = .nearest });
    defer qn.deinit();
    std.debug.print("  Min: {d:.1}, Max: {d:.1}, Range: {d:.1}, Median(nearest): {d:.1}\n", .{
        try lo.get(f64, &.{}), try hi.get(f64, &.{}),
        try r.get(f64, &.{}),  try qn.get(f64, &.{}),
    });
}
