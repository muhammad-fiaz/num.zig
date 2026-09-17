# CPU Parallel Execution API

Module: `@import("num").parallel`

---

## Configuration Options

Parallel operations accept an optional inline configuration structure:

```zig
.{
    .workers: ?usize = null,    // worker count, defaults to @min(cpu_count, 64)
    .chunkSize: ?usize = null,  // elements per chunk, defaults to count / workers
    .threshold: usize = 16384,  // minimum element count to spawn parallel threads
}
```

---

## Functions

### `run`
Executes a kernel function across range `[0, count)` partitioned across available CPU workers:

```zig
pub fn run(
    count: usize,
    context: anytype,
    comptime kernel: fn (@TypeOf(context), usize, usize) void,
    args: anytype, // optional inline configuration struct or .{}
) void;
```

#### Example

```zig
const Context = struct { slice: []f32 };
const ctx = Context{ .slice = data };

num.parallel.run(data.len, ctx, struct {
    fn kernel(c: Context, start: usize, end: usize) void {
        for (start..end) |i| c.slice[i] = @sqrt(c.slice[i]);
    }
}.kernel, .{
    .workers = 8,
    .chunkSize = 4096,
});
```

