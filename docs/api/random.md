# Random API Reference

The `random` module provides a random number generator and various probability distributions.

## Random Struct

The core of the module is the `Random` struct, which wraps a random number generator.

```zig
pub const Random = struct {
    // ... internal state ...
};
```

### init

Initialize a new Random number generator with a seed.

```zig
pub fn init(seed: u64) Random
```

## Distributions

### uniform

Generate a random float in the range [0, 1).

```zig
pub fn uniform(self: *Random, comptime T: type) T
```

### normal

Generate a random float from a normal distribution (mean=0, std=1).

```zig
pub fn normal(self: *Random, comptime T: type) T
```

### exponential

Generate a random float from an exponential distribution.

```zig
pub fn exponential(self: *Random, comptime T: type) T
```

### poisson

Generate a random integer from a Poisson distribution.

```zig
pub fn poisson(self: *Random, comptime T: type, lam: f64) T
```

### randint

Generate a random integer in the range [low, high).

```zig
pub fn randint(self: *Random, comptime T: type, low: T, high: T) T
```

## Array Operations

### shuffle

Shuffle an array in-place.

```zig
pub fn shuffle(self: *Random, comptime T: type, arr: NDArray(T)) void
```

### permutation

Return a permuted range or array.

```zig
pub fn permutation(self: *Random, allocator: Allocator, comptime T: type, n: usize) !NDArray(T)
```

### choice

Generates a random sample from a given 1-D array.

```zig
pub fn choice(self: *Random, allocator: Allocator, comptime T: type, a: NDArray(T), size: usize, replace: bool) !NDArray(T)
```