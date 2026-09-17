//! Demonstrates complex number construction and helpers.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const C128 = std.math.Complex(f64);

    const vals = [_]C128{ C128.init(3.0, 4.0), C128.init(1.0, -1.0) };
    var a = try num.fromSlice(allocator, C128, .{ .data = &vals, .shape = &.{2} });
    defer a.deinit();

    var c = try num.ops.conj(a);
    defer c.deinit();
    var re = try num.ops.real(a);
    defer re.deinit();
    var im = try num.ops.imag(a);
    defer im.deinit();
    var mag = try num.ops.magnitude(a);
    defer mag.deinit();
    var ph = try num.ops.phase(a);
    defer ph.deinit();

    std.debug.print("z[0] = {d:.1}+{d:.1}i\n", .{ (try a.asSlice(C128))[0].re, (try a.asSlice(C128))[0].im });
    std.debug.print("conj(z[0]) = {d:.1}+{d:.1}i\n", .{ (try c.asSlice(C128))[0].re, (try c.asSlice(C128))[0].im });
    std.debug.print("real = {any}\n", .{try re.asSlice(f64)});
    std.debug.print("imag = {any}\n", .{try im.asSlice(f64)});
    std.debug.print("|z[0]| = {d:.4} (expected 5.0)\n", .{try mag.get(f64, &.{0})});
    std.debug.print("arg(z[0]) = {d:.4}\n", .{try ph.get(f64, &.{0})});

    // Complex elementwise arithmetic preserves dtype.
    var sum = try num.ops.add(a, a, .{});
    defer sum.deinit();
    std.debug.print("z+z dtype={s} re[0]={d:.1}\n", .{ @tagName(sum.dtype), (try sum.asSlice(C128))[0].re });

    // Conjugate transpose of a 2x2 complex matrix.
    const mvals = [_]C128{
        C128.init(1.0, 1.0), C128.init(2.0, 0.0),
        C128.init(0.0, 3.0), C128.init(4.0, -1.0),
    };
    var m = try num.fromSlice(allocator, C128, .{ .data = &mvals, .shape = &.{ 2, 2 } });
    defer m.deinit();
    var h = try num.ops.conjTranspose(m);
    defer h.deinit();
    const h00 = try h.get(C128, &.{ 0, 0 });
    std.debug.print("conjTranspose[0,0] = {d:.1}+{d:.1}i\n", .{ h00.re, h00.im });
}
