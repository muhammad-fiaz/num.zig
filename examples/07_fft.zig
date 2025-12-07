const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== FFT Example ===\n\n", .{});

    var a = try num.NDArray(f32).init(allocator, &.{4});
    defer a.deinit();
    a.fill(1.0); // DC signal

    var fft_res = try num.fft.FFT.fft(allocator, &a);
    defer fft_res.deinit();

    std.debug.print("Input: [1, 1, 1, 1]\n", .{});
    std.debug.print("FFT Output:\n", .{});
    for (0..fft_res.shape[0]) |i| {
        const val = try fft_res.get(&.{i});
        std.debug.print("{d:.2} + {d:.2}i\n", .{ val.re, val.im });
    }

    var ifft_res = try num.fft.FFT.ifft(allocator, &fft_res);
    defer ifft_res.deinit();

    std.debug.print("\nIFFT Output (should match input):\n", .{});
    for (0..ifft_res.shape[0]) |i| {
        const val = try ifft_res.get(&.{i});
        std.debug.print("{d:.2} + {d:.2}i\n", .{ val.re, val.im });
    }

    // Sine wave
    std.debug.print("\n--- Sine Wave FFT ---\n", .{});
    var sine = try num.NDArray(f32).init(allocator, &.{8});
    defer sine.deinit();
    for (0..8) |i| {
        const t = @as(f32, @floatFromInt(i)) / 8.0;
        try sine.set(&.{i}, std.math.sin(2.0 * std.math.pi * t)); // 1 Hz
    }

    var sine_fft = try num.fft.FFT.fft(allocator, &sine);
    defer sine_fft.deinit();

    std.debug.print("Sine Wave FFT (Magnitude):\n", .{});
    for (0..sine_fft.shape[0]) |i| {
        const val = try sine_fft.get(&.{i});
        std.debug.print("{d}: {d:.2}\n", .{ i, val.magnitude() });
    }
}
