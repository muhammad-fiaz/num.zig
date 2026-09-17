# Random Number Generation (`num.random`)

The `num.random` module provides a seedable `Prng` wrapping Zig 0.16.0 `std.Random.DefaultPrng` (verified in the local `std`) plus distribution samplers reusing `std.Random` primitives (`float`, `floatNorm`, `intRangeLessThan`, `shuffle`). There is no shared global RNG state; pass an explicit `*Prng` for reproducibility or `null` for a thread-local counter-seeded fallback.

---

## 1. Initializing PRNG

```zig
var rng = num.random.Prng.init(42); // Seeded with integer 42
```

---

## 2. Uniform Distributions

### `uniform`
Draw samples uniformly distributed over $[low, high)$:
```zig
// Generate 100 random floats in range [0.0, 1.0)
var u = try num.random.uniform(allocator, .{
    .shape = &.{100},
    .low = 0.0,
    .high = 1.0,
    .dtype = .f64,
    .rng = &rng,
});
defer u.deinit();
```

### `rand`
Convenience uniform $[0.0, 1.0)$ helper:
```zig
var r = try num.random.rand(allocator, &.{100}, &rng);
defer r.deinit();
```

### `integers`
Draw discrete integers uniformly from low to high:
```zig
// Generate a 4x4 matrix of random integers from 1 to 10
var ints = try num.random.integers(allocator, .{
    .shape = &.{ 4, 4 },
    .low = 1,
    .high = 10,
    .dtype = .i64,
    .rng = &rng,
});
defer ints.deinit();
```

---

## 3. Normal (Gaussian) Distributions

### `normal`
Draw samples from a Gaussian distribution with mean $\mu$ (`loc`) and standard deviation $\sigma$ (`scale`), reusing `std.Random.floatNorm`:
```zig
var norm_dist = try num.random.normal(allocator, .{
    .shape = &.{1000},
    .loc = 0.0,
    .scale = 1.0,
    .dtype = .f64,
    .rng = &rng,
});
defer norm_dist.deinit();
```

### `randn`
Convenience standard normal (mean 0, std 1) helper:
```zig
var rn = try num.random.randn(allocator, &.{1000}, &rng);
defer rn.deinit();
```

---

## 4. Shuffling & Permutations

### `shuffle`
In-place Fisher-Yates shuffle along the first axis (contiguous 1D path reuses `std.Random.shuffle`):
```zig
try num.random.shuffle(&mut_arr, .{ .rng = &rng });
```

### `permutation`
Returns a newly permuted copy of an array:
```zig
var perm = try num.random.permutation(arr, .{ .rng = &rng });
defer perm.deinit();
```

### `choice`
Randomly sample elements from a 1D array with or without replacement:
```zig
var sample = try num.random.choice(allocator, source_arr, .{
    .size = 5,
    .replace = false,
    .rng = &rng,
});
defer sample.deinit();
```
