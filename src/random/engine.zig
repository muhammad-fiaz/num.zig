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

    /// Initializes a PRNG using system entropy.
    pub fn initDefault() Prng {
        const seed_val = std.crypto.random.int(u64);
        return Prng.init(seed_val);
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

var global_prng: ?Prng = null;

/// Returns a shared, lazily-initialized global PRNG.
pub fn getDefaultPrng() *Prng {
    if (global_prng == null) {
        global_prng = Prng.init(42);
    }
    return &global_prng.?;
}

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
