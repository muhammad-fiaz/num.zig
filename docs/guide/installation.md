# Installation & Package Configuration

`num.zig` is distributed as a standard Zig package and works with Zig 0.16.0.

> [!WARNING]
> Zig **0.15** is deprecated. New projects should use **Zig 0.16.0+** with **num.zig v0.0.3**.

---

## Method 1: Zig Fetch (Recommended)

**Latest Release (v0.0.3)**

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.3.tar.gz
```

**Previous Releases (v0.0.2, v0.0.1)**

```bash
zig fetch --save https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.2.tar.gz
```

---

## Method 2: Zig Fetch (Latest Development Build)

Use this for the latest development build from the `main` branch:

```bash
zig fetch --save git+https://github.com/muhammad-fiaz/num.zig.git
```

---

## Method 3: Manual `build.zig.zon` Configuration

```zig
.dependencies = .{
    .num = .{
        .url = "https://github.com/muhammad-fiaz/num.zig/archive/refs/tags/v0.0.3.tar.gz",
        .hash = "...", // Run `zig fetch --save <url>` to generate the hash automatically.
    },
},
```

---

## Method 4: Local Source Checkout

```bash
git clone https://github.com/muhammad-fiaz/num.zig.git
cd num.zig
zig build test
```

To use a local checkout from another project:

```zig
.dependencies = .{
    .num = .{
        .path = "../num.zig",
    },
},
```

---

## Wire into `build.zig`

Add the `num` module import to your executable or library in `build.zig`:

```zig
const num_dep = b.dependency("num", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("num", num_dep.module("num"));
```

---

## Verifying the Installation

Run the test suite and examples to confirm everything works:

```bash
zig build test
zig build run-array_creation
```
