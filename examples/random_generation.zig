//! Demonstrates PRNG engines and random distributions.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Seeded pseudo-random generator
    var rng = num.random.Prng.init(12345);

    // 2. Uniform random floats: f32 and f64
    var u_f32 = try num.random.uniform(allocator, .{
        .low = 0.0,
        .high = 1.0,
        .shape = &.{3},
        .dtype = .f32,
        .rng = &rng,
    });
    defer u_f32.deinit();

    std.debug.print("Uniform f32 in [0.0, 1.0):\n  [{d:.3}, {d:.3}, {d:.3}]\n", .{
        try u_f32.get(f32, &.{0}),
        try u_f32.get(f32, &.{1}),
        try u_f32.get(f32, &.{2}),
    });

    var u = try num.random.uniform(allocator, .{
        .low = 0.0,
        .high = 10.0,
        .shape = &.{ 2, 3 },
        .dtype = .f64,
        .rng = &rng,
    });
    defer u.deinit();

    std.debug.print("Uniform f64 in [0.0, 10.0) (2x3):\n", .{});
    for (0..2) |r| {
        std.debug.print("  row {d}: [{d:.2}, {d:.2}, {d:.2}]\n", .{
            r,
            try u.get(f64, &.{ r, 0 }),
            try u.get(f64, &.{ r, 1 }),
            try u.get(f64, &.{ r, 2 }),
        });
    }

    // 3. Normal distribution: f32 and f64
    var n_f32 = try num.random.normal(allocator, .{
        .loc = 0.0,
        .scale = 1.0,
        .shape = &.{3},
        .dtype = .f32,
        .rng = &rng,
    });
    defer n_f32.deinit();
    std.debug.print("Normal f32 (3 elements):\n  [{d:.3}, {d:.3}, {d:.3}]\n", .{
        try n_f32.get(f32, &.{0}),
        try n_f32.get(f32, &.{1}),
        try n_f32.get(f32, &.{2}),
    });

    var n = try num.random.normal(allocator, .{
        .loc = 0.0,
        .scale = 1.0,
        .shape = &.{4},
        .dtype = .f64,
        .rng = &rng,
    });
    defer n.deinit();

    std.debug.print("Normal f64 (4 elements):\n  [", .{});
    for (0..4) |i| {
        std.debug.print("{d:.3} ", .{try n.get(f64, &.{i})});
    }
    std.debug.print("]\n", .{});

    // 4. Random integers
    var ints = try num.random.integers(allocator, .{
        .low = 1,
        .high = 100,
        .shape = &.{5},
        .dtype = .i64,
        .rng = &rng,
    });
    defer ints.deinit();

    std.debug.print("Random integers [1..100):\n  [", .{});
    for (0..5) |i| {
        std.debug.print("{d} ", .{try ints.get(i64, &.{i})});
    }
    std.debug.print("]\n", .{});
}
