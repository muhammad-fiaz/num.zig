# Random Number Generation (`num.random`)

The `num.random` module provides high-quality pseudo-random number generation (PRNG) and distribution samplers built upon fast Xoroshiro / SplitMix engines.

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
var u = try num.random.uniform(allocator, f64, &rng, 0.0, 1.0, &.{100});
defer u.deinit();
```

### `integers`
Draw discrete integers uniformly from low to high:
```zig
// Generate a 4x4 matrix of random integers from 1 to 10
var ints = try num.random.integers(allocator, i32, &rng, 1, 10, &.{ 4, 4 });
defer ints.deinit();
```

---

## 3. Normal (Gaussian) & Other Distributions

### `normal`
Draw samples from a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$ (using Box-Muller transform):
```zig
var norm_dist = try num.random.normal(allocator, f64, &rng, 0.0, 1.0, &.{1000});
defer norm_dist.deinit();
```

### Additional Distributions
- **`exponential`**: Exponential rate parameter $\lambda$.
- **`poisson`**: Discrete events sampling.
- **`gamma`**: Shape and scale parameters.

---

## 4. Shuffling & Permutations

### `shuffle`
In-place Fisher-Yates shuffle along the first axis:
```zig
num.random.shuffle(f64, &rng, &mut_arr);
```

### `permutation`
Returns a newly permuted copy of an array:
```zig
var perm = try num.random.permutation(allocator, f64, &rng, &arr);
defer perm.deinit();
```

### `choice`
Randomly sample elements from a 1D array with or without replacement:
```zig
var sample = try num.random.choice(allocator, f64, &rng, &source_arr, 5, false);
defer sample.deinit();
```
