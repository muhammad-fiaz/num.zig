const std = @import("std");
const num = @import("num");
const NDArray = num.NDArray;
const Complex = std.math.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Complex Numbers Example ===\n\n", .{});

    // Create a complex array
    var a = try NDArray(Complex(f32)).init(allocator, &.{2});
    defer a.deinit(allocator);

    a.data[0] = Complex(f32).init(1.0, 2.0); // 1 + 2i
    a.data[1] = Complex(f32).init(3.0, 4.0); // 3 + 4i

    std.debug.print("Original Complex Array:\n", .{});
    for (a.data) |val| {
        std.debug.print("{d:.1} + {d:.1}i\n", .{ val.re, val.im });
    }

    // Real part
    var re = try num.complex.real(allocator, f32, a);
    defer re.deinit(allocator);
    std.debug.print("\nReal Part:\n", .{});
    for (re.data) |val| {
        std.debug.print("{d:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    // Imaginary part
    var im = try num.complex.imag(allocator, f32, a);
    defer im.deinit(allocator);
    std.debug.print("\nImaginary Part:\n", .{});
    for (im.data) |val| {
        std.debug.print("{d:.1} ", .{val});
    }
    std.debug.print("\n", .{});

    // Magnitude (Abs)
    var mag = try num.complex.abs(allocator, f32, a);
    defer mag.deinit(allocator);
    std.debug.print("\nMagnitude:\n", .{});
    for (mag.data) |val| {
        std.debug.print("{d:.4} ", .{val});
    }
    std.debug.print("\n", .{});

    // Conjugate
    var conj = try num.complex.conj(allocator, f32, a);
    defer conj.deinit(allocator);
    std.debug.print("\nConjugate:\n", .{});
    for (conj.data) |val| {
        std.debug.print("{d:.1} + {d:.1}i\n", .{ val.re, val.im });
    }
}
