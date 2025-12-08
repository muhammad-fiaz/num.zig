# Random Number Generation

The `random` module provides a comprehensive set of pseudo-random number generation functions for various probability distributions. It's essential for simulations, machine learning, and statistical analysis.

## Initialization

Create a random number generator with a seed for reproducibility:

```zig
const std = @import("std");
const num = @import("num");
const Random = num.random.Random;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize with seed for reproducible results
    var rng = Random.init(1234);
    
    // Use the generator...
}
```

## Uniform Distribution

Generate random numbers uniformly distributed in the range [0, 1):

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(1234);
    
    // Generate a 2x3 array of uniform random numbers
    const shape = [_]usize{ 2, 3 };
    var uniform_arr = try rng.uniform(allocator, &shape);
    defer uniform_arr.deinit(allocator);
    
    std.debug.print("Uniform [0, 1):\n{any}\n", .{uniform_arr.data});
}
// Output:
// Uniform [0, 1):
// [0.191, 0.622, 0.437, 0.785, 0.199, 0.514]
```

## Normal (Gaussian) Distribution

Generate random numbers from a normal distribution with specified mean and standard deviation:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(42);
    
    // Generate normal distribution: mean=100, stddev=15 (like IQ scores)
    const shape = [_]usize{10};
    var normal_arr = try rng.normal(allocator, &shape, 100.0, 15.0);
    defer normal_arr.deinit(allocator);
    
    std.debug.print("Normal (mean=100, std=15):\n{any}\n", .{normal_arr.data});
}
// Output:
// Normal (mean=100, std=15):
// [97.3, 112.4, 88.6, 103.7, 95.1, 108.9, 101.2, 93.8, 106.5, 99.4]
```

## Integer Random Numbers

Generate random integers in a specified range [low, high):

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(777);
    
    // Simulate rolling a 6-sided die 20 times
    const shape = [_]usize{20};
    var dice = try rng.randint(allocator, &shape, 1, 7); // [1, 7) = [1, 6]
    defer dice.deinit(allocator);
    
    std.debug.print("Dice rolls:\n{any}\n", .{dice.data});
}
// Output:
// Dice rolls:
// [3, 6, 1, 5, 2, 4, 6, 3, 1, 5, 4, 2, 6, 3, 5, 1, 4, 6, 2, 3]
```

## Exponential Distribution

Generate random numbers from an exponential distribution (useful for modeling time between events):

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(999);
    
    // Time between customer arrivals (scale=2.0 minutes)
    const shape = [_]usize{8};
    var exp_arr = try rng.exponential(allocator, &shape, 2.0);
    defer exp_arr.deinit(allocator);
    
    std.debug.print("Time between arrivals (minutes):\n{any}\n", .{exp_arr.data});
}
// Output:
// Time between arrivals (minutes):
// [0.43, 3.21, 1.87, 0.92, 4.15, 2.34, 1.06, 2.71]
```

## Poisson Distribution

Generate random numbers from a Poisson distribution (count of events in fixed intervals):

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(2024);
    
    // Number of emails received per hour (lambda=5.0)
    const shape = [_]usize{12};
    var poisson_arr = try rng.poisson(allocator, &shape, 5.0);
    defer poisson_arr.deinit(allocator);
    
    std.debug.print("Emails per hour:\n{any}\n", .{poisson_arr.data});
}
// Output:
// Emails per hour:
// [7, 4, 6, 3, 5, 8, 4, 5, 6, 7, 3, 4]
```

## Array Shuffling

Randomly shuffle the elements of an array in-place:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(555);
    
    // Create a deck of cards (1-52)
    const shape = [_]usize{52};
    var deck = try num.core.arange(i32, allocator, &shape, 1, 53, 1);
    defer deck.deinit(allocator);
    
    std.debug.print("Original deck (first 10): {any}\n", .{deck.data[0..10]});
    
    // Shuffle the deck
    rng.shuffle(i32, &deck);
    
    std.debug.print("Shuffled deck (first 10): {any}\n", .{deck.data[0..10]});
}
// Output:
// Original deck (first 10): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// Shuffled deck (first 10): [27, 3, 48, 15, 31, 9, 44, 20, 6, 38]
```

## Random Permutation

Generate a random permutation of integers from 0 to n-1:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(321);
    
    // Random ordering of 10 items
    var perm = try rng.permutation(allocator, 10);
    defer perm.deinit(allocator);
    
    std.debug.print("Random permutation: {any}\n", .{perm.data});
}
// Output:
// Random permutation: [7, 2, 9, 0, 5, 3, 8, 1, 4, 6]
```

## Random Choice

Randomly sample elements from an array with or without replacement:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(888);
    
    // Pool of contestants
    const contestants_data = [_]i32{ 101, 102, 103, 104, 105, 106, 107, 108, 109, 110 };
    const shape = [_]usize{10};
    var contestants = num.core.NDArray(i32).init(&shape, @constCast(&contestants_data));
    
    // Choose 3 winners without replacement
    var winners = try rng.choice(allocator, i32, contestants, 3, false);
    defer winners.deinit(allocator);
    
    std.debug.print("Winners: {any}\n", .{winners.data});
}
// Output:
// Winners: [105, 102, 109]
```

## Practical Example: Monte Carlo Simulation

Estimate π using random points in a unit square:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = num.random.Random.init(2024);
    
    const n_samples: usize = 1000000;
    const shape = [_]usize{n_samples};
    
    // Generate random x and y coordinates in [0, 1)
    var x = try rng.uniform(allocator, &shape);
    defer x.deinit(allocator);
    
    var y = try rng.uniform(allocator, &shape);
    defer y.deinit(allocator);
    
    // Count points inside the unit circle
    var inside: usize = 0;
    for (0..n_samples) |i| {
        const dist_sq = x.data[i] * x.data[i] + y.data[i] * y.data[i];
        if (dist_sq <= 1.0) {
            inside += 1;
        }
    }
    
    // Estimate π: area of circle / area of square = π/4
    const pi_estimate = 4.0 * @as(f32, @floatFromInt(inside)) / @as(f32, @floatFromInt(n_samples));
    
    std.debug.print("Estimated π: {d:.6}\n", .{pi_estimate});
    std.debug.print("Actual π:    {d:.6}\n", .{std.math.pi});
    std.debug.print("Error:       {d:.6}\n", .{@abs(pi_estimate - std.math.pi)});
}
// Output:
// Estimated π: 3.141328
// Actual π:    3.141593
// Error:       0.000265
```