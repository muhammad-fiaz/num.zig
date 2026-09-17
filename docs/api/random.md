# Random Number Generation API

Module: `@import("num").random`

---

## Types

### `Prng`
Fast 64-bit Xoshiro256** pseudo-random number generator.
```zig
pub const Prng = struct {
    pub fn init(seed: u64) Prng;
    pub fn random(self: *Prng) std.Random;
};
```

---

## Distribution Functions

All distribution functions accept an optional `rng: ?*Prng` — pass an explicit `Prng` for reproducibility, or `null` for system-entropy seeding (no shared global state).

```zig
pub fn uniform(allocator: std.mem.Allocator, options: struct { shape: []const usize = &.{}, low: f64 = 0.0, high: f64 = 1.0, dtype: DType = .f64, rng: ?*Prng = null }) !Array;

/// Convenience: uniform [0.0, 1.0), dtype=f64.
pub fn rand(allocator: std.mem.Allocator, shape: []const usize, rng: ?*Prng) !Array;

pub fn normal(allocator: std.mem.Allocator, options: struct { shape: []const usize = &.{}, loc: f64 = 0.0, scale: f64 = 1.0, dtype: DType = .f64, rng: ?*Prng = null }) !Array;

/// Convenience: standard normal (mean=0, std=1), dtype=f64.
pub fn randn(allocator: std.mem.Allocator, shape: []const usize, rng: ?*Prng) !Array;

pub fn integers(
    allocator: std.mem.Allocator,
    options: struct { shape: []const usize = &.{}, low: i64 = 0, high: i64 = 100, endpoint: bool = false, dtype: DType = .i64, rng: ?*Prng = null },
) !Array;
```

---

## Shuffling and Sampling

```zig
/// Randomly shuffle array elements in-place along axis 0.
pub fn shuffle(arr: *Array, rng: ?*Prng) void;

/// Return a shuffled copy of the array.
pub fn permutation(arr: Array, rng: ?*Prng) !Array;

/// Draw `size` elements from array `a`, with or without replacement.
pub fn choice(
    allocator: std.mem.Allocator,
    a: Array,
    size: usize,
    replace: bool,
    rng: ?*Prng,
) !Array;
```

