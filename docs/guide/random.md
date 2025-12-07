# Random Number Generation

The `num.random` module provides a `Random` struct for generating random numbers.

## Initialization

Initialize with a seed for reproducibility.

```zig
var rng = num.random.Random.init(42);
```

## Distributions

### Uniform

Generate values uniformly distributed in `[0, 1)`.

```zig
var u = try rng.uniform(allocator, &.{2, 3});
```

### Normal (Gaussian)

Generate values from a normal distribution with specified mean and standard deviation.

```zig
var n = try rng.normal(allocator, &.{2, 3}, 0.0, 1.0); // Mean 0, StdDev 1
```

### Integers

Generate random integers in `[low, high)`.

```zig
var i = try rng.randint(allocator, &.{5}, 0, 10);
```

## Shuffling

Shuffle an array along its first axis in-place.

```zig
rng.shuffle(f32, &arr);
```
