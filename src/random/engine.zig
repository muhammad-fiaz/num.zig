//! Pseudorandom number generator (PRNG) engine.
//!
//! Provides a deterministic, seedable PRNG state backed by Xoshiro256/DefaultPrng.

const std = @import("std");

pub const Prng = struct {
    engine: std.Random.DefaultPrng,

    /// Initializes a new PRNG with an explicit 64-bit seed.
    pub fn init(seed_val: u64) Prng {
        return Prng{
            .engine = std.Random.DefaultPrng.init(seed_val),
        };
    }

    /// Initializes a PRNG with a unique per-call seed without shared global state.
    /// Uses a thread-local counter mixed with a fixed base (no system-entropy
    /// dependency, verified against Zig 0.16.0 `std`).
    pub fn initDefault() Prng {
        const S = struct {
            threadlocal var counter: u64 = 0;
        };
        S.counter +%= 1;
        return Prng.init(0x9E3779B97F4A7C15 ^ (S.counter *% 0xBF58476D1CE4E5B9));
    }

    /// Re-seeds the PRNG with a new seed.
    pub fn seed(self: *Prng, seed_val: u64) void {
        self.engine = std.Random.DefaultPrng.init(seed_val);
    }

    /// Returns the standard library `Random` interface.
    pub fn random(self: *Prng) std.Random {
        return self.engine.random();
    }
};

test "Prng determinism and re-seeding" {
    var rng1 = Prng.init(12345);
    var rng2 = Prng.init(12345);

    const r1 = rng1.random();
    const r2 = rng2.random();

    for (0..10) |_| {
        try std.testing.expectEqual(r1.int(u32), r2.int(u32));
        try std.testing.expectEqual(r1.float(f64), r2.float(f64));
    }

    // Re-seeding produces the same sequence
    rng1.seed(12345);
    const r1_re = rng1.random();
    var rng3 = Prng.init(12345);
    const r3 = rng3.random();

    for (0..5) |_| {
        try std.testing.expectEqual(r1_re.int(u64), r3.int(u64));
    }
}
