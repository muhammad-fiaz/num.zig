# Bitwise Operations API

Module: `@import("num").ops`

Integer dtypes only (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`) with multidimensional broadcasting. Floating-point, boolean, and complex inputs return `DTypeError.UnsupportedDType`. Shift amounts come from the second operand with standard masking semantics.

---

## Logic & Shifts

```zig
pub fn bitwiseAnd(a: Array, b: Array) !Array; // a & b
pub fn bitwiseOr(a: Array, b: Array) !Array;  // a | b
pub fn bitwiseXor(a: Array, b: Array) !Array; // a ^ b
pub fn bitwiseNot(a: Array) !Array;           // ~a, owned contiguous copy
pub fn leftShift(a: Array, b: Array) !Array;  // a << b
pub fn rightShift(a: Array, b: Array) !Array; // a >> b (arithmetic if signed, logical if unsigned)
```

All binary operations broadcast their inputs and promote integer widths with `DType.promote`; results are newly owned arrays.

---

## Population Utilities

Per-element counts preserving the input dtype:

```zig
pub fn bitCount(a: Array) !Array; // alias: popcount, number of set bits
pub fn clz(a: Array) !Array;      // alias: leadingZeros, leading-zero count
pub fn ctz(a: Array) !Array;      // alias: trailingZeros, trailing-zero count
```

---

## Example

```zig
var and_res = try num.ops.bitwiseAnd(a, b);
defer and_res.deinit();

var shifted = try num.ops.leftShift(a, b);
defer shifted.deinit();

var counts = try num.ops.bitCount(a);
defer counts.deinit();
```
