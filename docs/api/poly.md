# Polynomials

The `poly` module provides polynomial operations.

## Functions

### `polyval`

Evaluate a polynomial at specific values.

```zig
pub fn polyval(allocator: Allocator, comptime T: type, p: NDArray(T), x: NDArray(T)) !NDArray(T)
```

**Parameters:**
- `p`: Polynomial coefficients (highest degree first).
- `x`: Values to evaluate at.

### `polyadd`

Add two polynomials.

```zig
pub fn polyadd(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

### `polysub`

Subtract two polynomials.

```zig
pub fn polysub(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

### `polymul`

Multiply two polynomials.

```zig
pub fn polymul(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

### `roots`

Return the roots of a polynomial.

```zig
pub fn roots(allocator: Allocator, comptime T: type, p: NDArray(T)) !NDArray(T)
```

### `polyder`

Return the derivative of a polynomial.

```zig
pub fn polyder(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize) !NDArray(T)
```

### `polyint`

Return the antiderivative (integral) of a polynomial.

```zig
pub fn polyint(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize, k: T) !NDArray(T)
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const poly = num.poly;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // p(x) = x^2 + 2x + 1
    var p = try num.NDArray(f32).init(allocator, &.{3}, &.{1.0, 2.0, 1.0});
    
    // Evaluate at x = 2
    var x = try num.NDArray(f32).init(allocator, &.{1}, &.{2.0});
    
    var y = try poly.polyval(allocator, f32, p, x);
    // y is 9.0
}
```

