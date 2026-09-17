const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1D signal: 8 samples
    const signal = [_]f64{ 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    var x = try num.fromSlice(allocator, f64, .{ .data = &signal, .shape = &.{8} });
    defer x.deinit();

    std.debug.print("Original Real Signal (8 samples):\n  [ ", .{});
    for (0..8) |i| {
        std.debug.print("{d:.1} ", .{try x.get(f64, &.{i})});
    }
    std.debug.print("]\n\n", .{});

    // Compute forward FFT
    var x_freq = try num.fft.fft(x, .{});
    defer x_freq.deinit();

    std.debug.print("FFT Spectrum (Complex128):\n", .{});
    for (0..8) |i| {
        const c = try x_freq.get(std.math.Complex(f64), &.{i});
        std.debug.print("  bin {d}: {d:6.3} + {d:6.3}i\n", .{ i, c.re, c.im });
    }

    // Compute inverse FFT to recover original signal
    var recovered = try num.fft.ifft(x_freq, .{});
    defer recovered.deinit();

    std.debug.print("\nRecovered Signal after IFFT:\n  [ ", .{});
    for (0..8) |i| {
        const c = try recovered.get(std.math.Complex(f64), &.{i});
        std.debug.print("{d:.1} ", .{c.re});
    }
    std.debug.print("]\n", .{});

    // Sample frequencies
    var freqs = try num.fft.fftfreq(allocator, 8, .{ .d = 0.1 });
    defer freqs.deinit();
    std.debug.print("FFT Frequencies [0]: {d:.2} Hz\n", .{try freqs.get(f64, &.{0})});

    // 2D FFT over the last two axes
    const img = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var m = try num.fromSlice(allocator, f64, .{ .data = &img, .shape = &.{ 2, 2 } });
    defer m.deinit();
    var f2 = try num.fft.fft2(m, .{});
    defer f2.deinit();
    var rt2 = try num.fft.ifft2(f2, .{});
    defer rt2.deinit();
    const back = try rt2.get(std.math.Complex(f64), &.{ 0, 0 });
    std.debug.print("FFT2 roundtrip [0,0]: {d:.3}\n", .{back.re});
}
