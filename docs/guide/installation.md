# Installation & Package Configuration

`num.zig` is distributed as a standard Zig package and works with Zig 0.16.0.

---

## 1. Adding to `build.zig.zon`

In your project directory, fetch `num.zig` directly:

```bash
zig fetch --save git+https://github.com/muhammad-fiaz/num.zig.git
```

This will automatically populate your `build.zig.zon`:

```zig
.{
    .name = .my_project,
    .version = "0.1.0",
    .fingerprint = 0x12345678,
    .dependencies = .{
        .num = .{
            .url = "git+https://github.com/muhammad-fiaz/num.zig.git#<commit-hash>",
            .hash = "...",
        },
    },
    .paths = .{""},
}
```

---

## 2. Importing into `build.zig`

Add the `num` module import to your executable or library in `build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const num_dep = b.dependency("num", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "my_app",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.root_module.addImport("num", num_dep.module("num"));
    b.installArtifact(exe);
}
```

---

## 3. Verifying Local Installation

You can clone and test `num.zig` directly:

```bash
git clone https://github.com/muhammad-fiaz/num.zig.git
cd num.zig
zig build test
zig build test-all
```
