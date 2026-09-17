# Comparisons & Logic API

Module: `@import("num").ops`

---

## Elementwise Comparisons

All comparison functions return a boolean `Array` (`dtype = .bool`) and support broadcasting.
Inputs are passed by value — no allocator or type parameter is required.

```zig
pub fn equal(a: Array, b: Array) !Array;
pub fn notEqual(a: Array, b: Array) !Array;
pub fn less(a: Array, b: Array) !Array;
pub fn lessEqual(a: Array, b: Array) !Array;
pub fn greater(a: Array, b: Array) !Array;
pub fn greaterEqual(a: Array, b: Array) !Array;
```

---

## Logical Operations

```zig
pub fn logicalAnd(a: Array, b: Array) !Array;
pub fn logicalOr(a: Array, b: Array) !Array;
pub fn logicalXor(a: Array, b: Array) !Array;
pub fn logicalNot(a: Array) !Array;
```

---

## Floating-Point Predicates

Returns a boolean Array where each element indicates the predicate holds.

```zig
pub fn isNaN(a: Array) !Array;
pub fn isInf(a: Array) !Array;
pub fn isFinite(a: Array) !Array;
```

---

## Approximate Float Equality

```zig
pub fn isClose(
    a: Array,
    b: Array,
    options: struct { rtol: f64 = 1e-5, atol: f64 = 1e-8, equalNan: bool = false },
) !Array;

pub fn allClose(
    a: Array,
    b: Array,
    options: struct { rtol: f64 = 1e-5, atol: f64 = 1e-8, equalNan: bool = false },
) !bool;
```

