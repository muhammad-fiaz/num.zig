# NZIG v1.0 Binary Format & Text I/O

`num.zig` provides both a binary serialization format (`NZIG v1.0`) and delimited plain-text I/O (`savetxt` / `loadtxt`).

---

## 1. The NZIG v1.0 Binary Format

The `NZIG` format is a fast, cross-platform binary serialization standard for N-dimensional numerical arrays.

### Header Specification

The binary header is exactly 128 bytes, 64-byte aligned, using `std.mem.writeInt` / `std.mem.readInt` little-endian encoding (verified against Zig 0.16.0 `std`):

| Offset | Size | Description |
| :--- | :--- | :--- |
| 0..7 | 8 bytes | Magic identifier (`0x89`, `N`, `Z`, `I`, `G`, `1`, `0`, `\n`) |
| 8..15 | 8 bytes | Specification version, flags, dtype id, rank (validated; unsupported rejected) |
| 16..23 | 8 bytes | Reserved / flags area (validated) |
| 24..31 | 8 bytes | Payload size in bytes, little-endian `u64` (validated against shape × dtype size with checked `std.math.mul`) |
| 32..95 | 64 bytes | Up to 8 dimension sizes as little-endian `u64` |
| 96..127 | 32 bytes | Reserved padding |

The raw, contiguous numerical payload directly succeeds the header block. Deserialization validates magic, version, dtype id, rank, dimensions, payload size, truncation, and trailing bytes; endianness conversion is applied when required.

---

## 2. Saving and Loading Binary Files

```zig
// Save to file
try num.save(allocator, "weights.nzig", tensor);

// Load from file
var loaded_tensor = try num.load(allocator, "weights.nzig");
defer loaded_tensor.deinit();
```

---

## 3. Delimited Text I/O (`savetxt` & `loadtxt`)

Read and write matrices from CSV, TSV, or whitespace-delimited text files:

### `savetxt`
```zig
try num.io.savetxt(allocator, matrix, "data.csv", .{
    .delimiter = ",",
    .header = "x,y,z",
});
```

### `loadtxt`
```zig
var parsed = try num.io.loadtxt(allocator, "data.csv", .{
    .delimiter = ",",
    .skipRows = 1,
});
defer parsed.deinit();
```
