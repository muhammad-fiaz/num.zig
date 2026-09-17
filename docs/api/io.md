# Serialization & I/O API

Module: `@import("num").io`

---

## NZIG v1.0 Binary I/O

Native binary format with a deterministic 128-byte header, stable dtype IDs, and little-endian payload.

```zig
/// Write an Array to a `.nzig` file at the given path.
pub fn writeFile(allocator: std.mem.Allocator, file_path: []const u8, arr: Array) !void;

/// Read an Array from a `.nzig` file at the given path.
pub fn readFile(allocator: std.mem.Allocator, file_path: []const u8) !Array;

/// Serialize an Array into any writer (stream I/O, buffers, sockets).
pub fn writeToStream(arr: Array, writer: anytype) !void;

/// Deserialize an Array from any reader.
pub fn readFromStream(allocator: std.mem.Allocator, reader: anytype) !Array;
```

Top-level convenience aliases:

```zig
pub const save = io.writeFile;
pub const load = io.readFile;
```

---

## Delimited Text I/O

```zig
pub fn savetxt(
    allocator: std.mem.Allocator,
    arr: Array,
    file_path: []const u8,
    options: struct { delimiter: []const u8 = ",", header: []const u8 = "", footer: []const u8 = "" },
) !void;

pub fn loadtxt(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    options: struct { delimiter: []const u8 = ",", skipRows: usize = 0, dtype: DType = .f64 },
) !Array;
```

---

## Utilities

```zig
/// In-memory byte stream for testing and embedding.
pub const MemoryStream = io.MemoryStream;

/// Format an Array to a writer for human-readable display.
pub fn formatArray(arr: Array, writer: anytype) !void;
```

