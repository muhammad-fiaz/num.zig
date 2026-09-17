# Data Types (DTypes) & Type Promotion

`num.zig` provides a comprehensive, strongly typed type system supporting 14 distinct numerical data types, including signed/unsigned integers, IEEE floating point numbers, boolean flags, and complex numbers.

---

## 1. Supported DTypes

The `num.DType` enum identifies data types across runtime and compile-time operations:

| DType Enum | Zig Native Type | Description | Size (Bytes) |
| :--- | :--- | :--- | :--- |
| `.bool` | `bool` | Boolean flag (`true` or `false`) | 1 |
| `.i8` | `i8` | 8-bit signed integer | 1 |
| `.i16` | `i16` | 16-bit signed integer | 2 |
| `.i32` | `i32` | 32-bit signed integer | 4 |
| `.i64` | `i64` | 64-bit signed integer | 8 |
| `.u8` | `u8` | 8-bit unsigned integer / raw byte | 1 |
| `.u16` | `u16` | 16-bit unsigned integer | 2 |
| `.u32` | `u32` | 32-bit unsigned integer | 4 |
| `.u64` | `u64` | 64-bit unsigned integer | 8 |
| `.f16` | `f16` | Half precision floating point | 2 |
| `.f32` | `f32` | Single precision floating point | 4 |
| `.f64` | `f64` | Double precision floating point | 8 |
| `.c64` | `Complex(f32)` | Single-precision complex (real, imag) | 8 |
| `.c128` | `Complex(f64)` | Double-precision complex (real, imag) | 16 |

---

## 2. Compile-Time Mapping & Introspection

`num.zig` provides helper utilities to translate between Zig types and `num.DType`:

```zig
const dt = num.DType.fromType(f64); // returns num.DType.f64
const size = dt.sizeOf();           // returns 8
const is_float = dt.isFloat();      // returns true
const is_int = dt.isInteger();      // returns false
```

---

## 3. Type Promotion Rules

When performing mixed binary operations (e.g. adding an `i32` array and an `f32` array), `num.zig` calculates the common promoted type according to standard mathematical rules:

```zig
const promoted = num.DType.promote(.i32, .f32); // returns .f64
```

### Hierarchy Lattice
1. If either operand is complex (`c128`), result promotes to `c128`.
2. If either operand is `c64`, promotes to `c64` (or `c128` if paired with `f64`).
3. Floating point operands dominate integer operands.
4. Larger bit widths dominate smaller bit widths.
5. Mixed signed/unsigned integers promote to the next wider signed type to prevent overflow.

---

## 4. Casting and Conversion

Convert an array between different types using `astype`:

```zig
var int_arr = try num.fromSlice(allocator, i32, .{
    .data = &[_]i32{ 1, 2, 3, 4 },
    .shape = &.{4},
});
defer int_arr.deinit();

// Cast to f64
var float_arr = try int_arr.astype(f64);
defer float_arr.deinit();
```
