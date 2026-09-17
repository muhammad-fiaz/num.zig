# Core Array & Memory API

Module: `@import("num")`

---

## Types

### `Array`
The central N-dimensional numerical container.
```zig
pub const Array = struct {
    allocator: std.mem.Allocator,
    dtype: DType,
    shape: Shape,
    strides: Strides,
    buffer: Buffer,

    pub fn deinit(self: *Array) void;
    pub fn data(self: *const Array, comptime T: type) []T;
    pub fn get(self: *const Array, comptime T: type, indices: []const usize) !T;
    pub fn getItem(self: *const Array, comptime T: type, indices: []const usize) !T; // direct alias to get
    pub fn set(self: *Array, comptime T: type, indices: []const usize, val: T) !void;
    pub fn fill(self: *Array, comptime T: type, val: T) void;
    pub fn take(self: *const Array, indices: *const Array, options: struct { axis: ?usize = null }) !Array;
    pub fn put(self: *Array, indices: *const Array, values: *const Array) !void;
    pub fn clone(self: *const Array) !Array;
    pub fn astype(self: *const Array, comptime T: type) !Array;
    pub fn isContiguous(self: *const Array) bool;
    pub fn reshapeInPlace(self: *Array, new_shape: []const usize) !void;
    pub fn broadcastTo(self: *const Array, new_shape: []const usize) !Array;
    pub fn slice(self: *const Array, slices: []const SliceSpec) !Array;
};
```

### `Shape`
Represents tensor dimensions up to rank 8 inline:
```zig
pub const Shape = struct {
    dims: [8]usize,
    rank: usize,

    pub fn init(slice: []const usize) Shape;
    pub fn totalElements(self: Shape) usize;
    pub fn slice(self: *const Shape) []const usize;
};
```

### `Strides`
Byte or element offsets per dimension:
```zig
pub const Strides = struct {
    values: [8]usize,
    rank: usize,

    pub fn fromShape(shape: Shape) Strides;
    pub fn offset(self: Strides, indices: []const usize) usize;
};
```

### `SliceSpec`
Specification for sub-array extraction:
```zig
pub const SliceSpec = struct {
    start: usize = 0,
    end: ?usize = null,
    step: usize = 1,
};
```

---

## Creation Functions

### `fromSlice`
Creates an `Array` initialized with data from a slice:
```zig
pub fn fromSlice(
    allocator: std.mem.Allocator,
    comptime T: type,
    options: struct {
        data: []const T,
        shape: []const usize,
    },
) !Array;
```

### `zeros`
Initializes an array filled with zeros:
```zig
pub fn zeros(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) !Array;
```

### `ones`
Initializes an array filled with ones:
```zig
pub fn ones(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) !Array;
```

### `full`
Initializes an array filled with a constant scalar:
```zig
pub fn full(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        value: anytype,
        dtype: ?DType = null,
        order: Order = .c,
    },
) !Array;
```

### `empty`
Allocates an uninitialized array buffer:
```zig
pub fn empty(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) !Array;
```

### `arange`
Generates half-open range `[start, stop)`:
```zig
pub fn arange(
    allocator: std.mem.Allocator,
    options: struct {
        start: anytype = 0,
        stop: anytype,
        step: anytype = 1,
        dtype: ?DType = null,
    },
) !Array;
```

### `linspace`
Generates `num` evenly spaced values over `[start, stop]`:
```zig
pub fn linspace(
    allocator: std.mem.Allocator,
    options: struct {
        start: anytype,
        stop: anytype,
        num: usize = 50,
        endpoint: bool = true,
        dtype: DType = .f64,
    },
) !Array;
```

### `logspace`
Generates numbers spaced evenly on a log scale:
```zig
pub fn logspace(
    allocator: std.mem.Allocator,
    options: struct {
        start: anytype,
        stop: anytype,
        num: usize = 50,
        base: f64 = 10.0,
        endpoint: bool = true,
        dtype: DType = .f64,
    },
) !Array;
```

### `geomspace`
Generates numbers spaced evenly on a geometric progression:
```zig
pub fn geomspace(
    allocator: std.mem.Allocator,
    options: struct {
        start: anytype,
        stop: anytype,
        num: usize = 50,
        endpoint: bool = true,
        dtype: DType = .f64,
    },
) !Array;
```

### `eye`
Creates a 2D matrix with ones on the diagonal:
```zig
pub fn eye(
    allocator: std.mem.Allocator,
    options: struct {
        n: usize,
        m: ?usize = null,
        k: isize = 0,
        dtype: DType = .f64,
    },
) !Array;
```

### `identity`
Creates an $N \times N$ identity matrix:
```zig
pub fn identity(
    allocator: std.mem.Allocator,
    options: struct {
        n: usize,
        dtype: DType = .f64,
    },
) !Array;
```

### `diag`
Extracts a diagonal or constructs a diagonal 2D matrix:
```zig
pub fn diag(
    allocator: std.mem.Allocator,
    v: Array,
    options: struct {
        k: isize = 0,
    },
) !Array;
```

