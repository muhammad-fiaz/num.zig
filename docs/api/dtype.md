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

    pub fn toType(comptime self: DType) type;
    pub fn fromType(comptime T: type) DType;
    pub fn sizeOf(self: DType) usize;
    pub fn alignmentOf(self: DType) usize;
    pub fn isFloat(self: DType) bool;
    pub fn isInteger(self: DType) bool;
    pub fn isSigned(self: DType) bool;
    pub fn isUnsigned(self: DType) bool;
    pub fn isComplex(self: DType) bool;
    pub fn isBoolean(self: DType) bool;
    pub fn promote(a: DType, b: DType) DType;
    pub fn castValue(comptime DstT: type, comptime SrcT: type, val: SrcT) DstT;
};
```

Complex element types reuse Zig 0.16.0 `std.math.Complex(f32)` (`c64`) and `std.math.Complex(f64)` (`c128`), verified in the local `std`.

---

## Functions

### `promote`
Determines the common promoted type between two data types:
```zig
pub fn promote(a: DType, b: DType) DType;
```
