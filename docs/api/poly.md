# Polynomial API Reference

The `poly` module provides functions for polynomial arithmetic and evaluation.

## Evaluation

### polyval

Evaluate a polynomial at specific values.

```zig
pub fn polyval(allocator: Allocator, comptime T: type, p: NDArray(T), x: NDArray(T)) !NDArray(T)
```

## Arithmetic

### polyadd

Add two polynomials.

```zig
pub fn polyadd(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

### polysub

Subtract two polynomials.

```zig
pub fn polysub(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

### polymul

Multiply two polynomials.

```zig
pub fn polymul(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T)
```

## Calculus

### polyder

Return the derivative of the specified order of a polynomial.

```zig
pub fn polyder(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize) !NDArray(T)
```

### polyint

Return an antiderivative (indefinite integral) of a polynomial.

```zig
pub fn polyint(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize, k: T) !NDArray(T)
```

## Roots

### roots

Return the roots of a polynomial with coefficients given in p.

```zig
pub fn roots(allocator: Allocator, comptime T: type, p: NDArray(T)) !NDArray(T)
```