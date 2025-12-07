# Random Sampling

The `random` module provides random number generation.

## Random

Wrapper around Zig's PRNG.

### `init`

Initialize with a seed.

```zig
pub fn init(seed: u64) Random
```

### `uniform`

Generate uniform random numbers [0, 1).

```zig
pub fn uniform(self: *Random, allocator: Allocator, shape: []const usize) !NDArray(f32)
```

### `normal`

Generate normal random numbers.

```zig
pub fn normal(self: *Random, allocator: Allocator, shape: []const usize, mean: f32, stddev: f32) !NDArray(f32)
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const Random = num.random.Random;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var rng = Random.init(42);
    var arr = try rng.uniform(allocator, &.{2, 2});
    defer arr.deinit();
}
```

