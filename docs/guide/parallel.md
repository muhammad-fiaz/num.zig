# CPU Parallel Execution

`num.zig` features a multi-threaded parallel execution subsystem engineered specifically for compute-intensive numerical workflows.

---

## 1. Automatic Dynamic Scheduling

For operations spanning thousands or millions of elements, `num.zig` dynamically chooses between sequential execution and thread pool scheduling:

- **Cache-Conscious Thresholds**: Small operations that easily fit in L1/L2 cache are kept sequential to eliminate thread synchronization overhead.
- **Nested Parallelism Guard**: If an algorithm recursively calls a parallel subroutine, `num.zig` automatically switches nested iterations to sequential mode to prevent thread contention.

---

## 2. Low-Level Worker Scheduling (`run`)

You can execute arbitrary closures or functions across an index range in parallel:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var a = try num.full(allocator, .{ .shape = &.{1_000_000}, .value = @as(f32, 16.0) });
    defer a.deinit();

    const Context = struct { arr: num.Array };
    const ctx = Context{ .arr = a };

    // Partition 1,000,000 elements across available CPU cores
    num.parallel.run(1_000_000, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const val = c.arr.get(f32, &.{i}) catch unreachable;
                c.arr.set(f32, &.{i}, @sqrt(val)) catch unreachable;
            }
        }
    }.kernel, .{
        .workers = 8,
        .chunkSize = 8192,
        .threshold = 10000,
    });
}
```

---

## 3. Inline Configuration Options

Customize thread pool behaviors using inline optional fields:

```zig
num.parallel.run(count, ctx, kernel, .{
    .workers = 8,        // null defaults to std.Thread.getCpuCount()
    .chunkSize = 4096,   // defaults to count / workers
    .threshold = 16384,  // workloads below threshold run sequentially
});
```
