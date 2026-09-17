# DType & Type Promotion API

Module: `@import("num")`

---

## Enums

### `DType`
Enumeration of supported numerical types.
```zig
pub const DType = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    c64,
    c128,

    pub fn size(self: DType) usize;
    pub fn isFloat(self: DType) bool;
    pub fn isInt(self: DType) bool;
    pub fn isSigned(self: DType) bool;
    pub fn isComplex(self: DType) bool;
    pub fn fromType(comptime T: type) DType;
};
```

---

## Functions

### `promoteDTypes`
Determines the common promoted type between two data types:
```zig
pub fn promoteDTypes(a: DType, b: DType) DType;
```

### `Complex`
Generic complex number representation:
```zig
pub fn Complex(comptime T: type) type {
    return struct {
        re: T,
        im: T,
    };
}
```
