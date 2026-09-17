# Core Array & Memory API

Module: `@import("num")`

---

## Types

### `Array`
The central N-dimensional numerical container with SBO shape/strides, explicit ownership, and strided views.
```zig
pub const Array = struct {
    allocator: std.mem.Allocator,
    dtype: DType,
    ndim: u8,
    shape_dims: [MAX_RANK]usize,
    stride_vals: [MAX_RANK]isize,
    flags: ArrayFlags,       // ownsData, isCContiguous, isFContiguous
    root_buffer: ?Buffer,

    pub fn deinit(self: *Array) void;                                     // no-op for views
    pub fn view(self: Array) Array;                                      // non-owning borrow; caller must ensure lifetime
    pub fn clone(self: Array) !Array;                                    // deep contiguous copy; alias: copy
    pub fn asContiguous(self: Array) !Array;                             // owned contiguous copy
    pub fn astype(self: Array, target: DType) !Array;                    // dtype-converting copy
    pub fn shapeSlice(self: *const Array) []const usize;
    pub fn stridesSlice(self: *const Array) []const isize;
    pub fn elementCount(self: Array) usize;
    pub fn byteCount(self: Array) usize;
    pub fn isContiguous(self: Array) bool;
    pub fn get(self: Array, comptime T: type, indices: []const usize) !T; // e.g. get(f32, &.{0}); DTypeMismatch on wrong T
    pub fn set(self: Array, comptime T: type, indices: []const usize, value: T) !void; // works on strided views
    pub fn item(self: Array, comptime T: type) !T;
    pub fn getAsFloat(self: Array, indices: []const usize) !f64;
    pub fn setFromFloat(self: Array, indices: []const usize, val: f64) !void;
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

### `scalar`
Creates a 0D array holding a single constant value (reuses `full`):
```zig
// options.value is any scalar; options.dtype is optional
pub fn scalar(allocator: std.mem.Allocator, options: anytype) !Array;
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

