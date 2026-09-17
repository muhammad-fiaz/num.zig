//! Demonstrates integer bitwise operations with broadcasting.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const a_vals = [_]i32{ 0b1100, 0b1010, 1, 8 };
    const b_vals = [_]i32{ 0b1010, 0b1100, 2, 1 };
    var a = try num.fromSlice(allocator, i32, .{ .data = &a_vals, .shape = &.{4} });
    defer a.deinit();
    var b = try num.fromSlice(allocator, i32, .{ .data = &b_vals, .shape = &.{4} });
    defer b.deinit();

    var and_res = try num.ops.bitwiseAnd(a, b);
    defer and_res.deinit();
    var or_res = try num.ops.bitwiseOr(a, b);
    defer or_res.deinit();
    var xor_res = try num.ops.bitwiseXor(a, b);
    defer xor_res.deinit();
    var shl = try num.ops.leftShift(a, b);
    defer shl.deinit();
    var shr = try num.ops.rightShift(a, b);
    defer shr.deinit();
    var not_a = try num.ops.bitwiseNot(a);
    defer not_a.deinit();
    var pc = try num.ops.bitCount(a);
    defer pc.deinit();

    std.debug.print("a & b = {any}\n", .{try and_res.asSlice(i32)});
    std.debug.print("a | b = {any}\n", .{try or_res.asSlice(i32)});
    std.debug.print("a ^ b = {any}\n", .{try xor_res.asSlice(i32)});
    std.debug.print("a << b[0]={d}: {d}\n", .{ try b.get(i32, &.{0}), try shl.get(i32, &.{0}) });
    std.debug.print("a >> b[0]={d}: {d}\n", .{ try b.get(i32, &.{0}), try shr.get(i32, &.{0}) });
    std.debug.print("~a[0]={d}: {d}\n", .{ try a.get(i32, &.{0}), try not_a.get(i32, &.{0}) });
    std.debug.print("popcount(a[0]={d})={d}\n", .{ try a.get(i32, &.{0}), try pc.get(i32, &.{0}) });

    // Scalar broadcast: mask a vector with 0xFF.
    const u_vals = [_]u8{ 0xF0, 0x0F };
    var u = try num.fromSlice(allocator, u8, .{ .data = &u_vals, .shape = &.{2} });
    defer u.deinit();
    const m_vals = [_]u8{0xFF};
    var mask = try num.fromSlice(allocator, u8, .{ .data = &m_vals, .shape = &.{} });
    defer mask.deinit();
    var masked = try num.ops.bitwiseAnd(u, mask);
    defer masked.deinit();
    std.debug.print("masked = {any}\n", .{try masked.asSlice(u8)});
}
