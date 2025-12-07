# Input / Output

The `num.io` module allows saving and loading arrays to disk.

## Format

The format is a simple binary format:
- Magic Header: "NUMZIG" (6 bytes)
- Version: 1 byte
- Rank: 1 byte
- Shape: `rank * 8` bytes (u64 little endian)
- Data: Raw bytes (row-major)

## Saving

```zig
try num.io.save(f32, arr, "data.num");
```

## Loading

```zig
var loaded = try num.io.load(allocator, f32, "data.num");
defer loaded.deinit();
```
