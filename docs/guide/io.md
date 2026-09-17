# NZIG v1.0 Binary Format & Text I/O

`num.zig` provides both a binary serialization format (`NZIG v1.0`) and delimited plain-text I/O (`savetxt` / `loadtxt`).

---

## 1. The NZIG v1.0 Binary Format

The `NZIG` format is a fast, cross-platform binary serialization standard for N-dimensional numerical arrays.

### Header Specification

The binary header consists of a fixed 64-byte block:

| Offset | Size | Type | Description |
| :--- | :--- | :--- | :--- |
| 0..3 | 4 bytes | `[4]u8` | Magic identifier `\x93NZIG` |
| 4..5 | 2 bytes | `u8, u8` | Major version (1), Minor version (0) |
| 6..7 | 2 bytes | `u16` (LE) | Header total length |
| 8..9 | 2 bytes | `u16` (LE) | DType identifier enum |
| 10 | 1 byte | `u8` | Endianness flag (`0` = Little Endian, `1` = Big Endian) |
| 11 | 1 byte | `u8` | Memory order (`0` = C-contiguous, `1` = Fortran) |
| 12..15 | 4 bytes | `u32` (LE) | Rank (number of dimensions) |
| 16..47 | 32 bytes | `[8]u32` (LE) | Dimension sizes |
| 48..63 | 16 bytes | `[16]u8` | Reserved padding |

The raw, contiguous numerical payload directly succeeds the header block.

---

## 2. Saving and Loading Binary Files

```zig
// Save to file
try num.io.save(allocator, "weights.nzig", &tensor);

// Load from file
var loaded_tensor = try num.io.load(allocator, "weights.nzig");
defer loaded_tensor.deinit();
```

---

## 3. Delimited Text I/O (`savetxt` & `loadtxt`)

Read and write matrices from CSV, TSV, or whitespace-delimited text files:

### `savetxt`
```zig
try num.io.savetxt(allocator, f64, "data.csv", &matrix, .{
    .delimiter = ',',
    .header = "x,y,z",
    .fmt = "%.4f",
});
```

### `loadtxt`
```zig
var parsed = try num.io.loadtxt(allocator, f64, "data.csv", .{
    .delimiter = ',',
    .skip_rows = 1,
});
defer parsed.deinit();
```
