//! Core Array structure, views, and creation factories.
//!
//! Provides the primary N-dimensional array type with Small Buffer Optimization (SBO),
//! explicit allocator discipline, zero-allocation views, and hardware-aligned memory storage.

const std = @import("std");
const DType = @import("dtype.zig").DType;
const Shape = @import("shape.zig").Shape;
const Strides = @import("shape.zig").Strides;
const Order = @import("shape.zig").Order;
const MAX_RANK = @import("shape.zig").MAX_RANK;
const Buffer = @import("buffer.zig").Buffer;
const ShapeError = @import("error.zig").ShapeError;
const IndexError = @import("error.zig").IndexError;
const DTypeError = @import("error.zig").DTypeError;

/// Flags describing memory layout and ownership semantics.
pub const ArrayFlags = struct {
    /// True if this array owns its underlying memory and must free it upon deinit.
    ownsData: bool = true,
    /// True if memory layout is C-contiguous (row-major).
    isCContiguous: bool = true,
    /// True if memory layout is Fortran-contiguous (column-major).
    isFContiguous: bool = false,
};

/// Primary N-dimensional array struct.
pub const Array = struct {
    /// Client-supplied allocator used for lifecycle operations.
    allocator: std.mem.Allocator,
    /// Pointer to the first element of this array view.
    data_ptr: [*]u8,
    /// Element scalar data type.
    dtype: DType,
    /// Number of dimensions (0 for scalars, up to MAX_RANK).
    ndim: u8,
    /// Dimension sizes.
    shape_dims: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    /// Dimension strides (in element counts).
    stride_vals: [MAX_RANK]isize = [_]isize{0} ** MAX_RANK,
    /// Memory flags.
    flags: ArrayFlags = .{},
    /// Owning root buffer (non-null only if ownsData is true).
    root_buffer: ?Buffer = null,

    /// Releases memory if this array owns its storage. Safe no-op for views.
    pub fn deinit(self: *Array) void {
        if (self.flags.ownsData and self.root_buffer != null) {
            self.root_buffer.?.deinit();
            self.root_buffer = null;
        }
        self.flags.ownsData = false;
    }

    /// Returns the shape as a Shape struct.
    pub fn shape(self: *const Array) Shape {
        var s = Shape{ .ndim = self.ndim };
        for (0..self.ndim) |i| {
            s.dims[i] = self.shape_dims[i];
        }
        return s;
    }

    /// Returns the dimension sizes as a slice.
    pub fn shapeSlice(self: *const Array) []const usize {
        return self.shape_dims[0..self.ndim];
    }

    /// Returns the strides as a Strides struct.
    pub fn strides(self: *const Array) Strides {
        var st = Strides{ .ndim = self.ndim };
        for (0..self.ndim) |i| {
            st.values[i] = self.stride_vals[i];
        }
        return st;
    }

    /// Returns the dimension strides as a slice.
    pub fn stridesSlice(self: *const Array) []const isize {
        return self.stride_vals[0..self.ndim];
    }

    /// Total number of elements across all dimensions.
    pub fn elementCount(self: Array) usize {
        return self.shape().elementCount();
    }

    /// Total memory size in bytes.
    pub fn byteCount(self: Array) usize {
        return self.elementCount() * self.dtype.sizeOf();
    }

    /// Returns true if the array data is stored contiguously in C-order.
    pub fn isContiguous(self: Array) bool {
        return self.flags.isCContiguous;
    }

    /// Creates a non-owning borrow/view into this array with identical shape and strides.
    pub fn view(self: Array) Array {
        var v = self;
        v.flags.ownsData = false;
        v.root_buffer = null;
        return v;
    }

    /// Duplicates the array into a newly allocated, contiguous owned copy.
    pub fn clone(self: Array) (ShapeError || std.mem.Allocator.Error)!Array {
        var copy = try zeros(self.allocator, .{
            .shape = self.shapeSlice(),
            .dtype = self.dtype,
        });

        if (self.isContiguous()) {
            @memcpy(copy.data_ptr[0..self.byteCount()], self.data_ptr[0..self.byteCount()]);
        } else {
            const NdIterator = @import("iterator.zig").NdIterator;
            var it = NdIterator.init(self.shape(), self.strides());
            const elem_sz = self.dtype.sizeOf();
            var out_offset: usize = 0;

            while (it.next()) |it_item| {
                const src_ptr = self.data_ptr + @as(usize, @intCast(@as(isize, @intCast(0)) + it_item.offset)) * elem_sz;
                const dst_ptr = copy.data_ptr + out_offset * elem_sz;
                @memcpy(dst_ptr[0..elem_sz], src_ptr[0..elem_sz]);
                out_offset += 1;
            }
        }

        return copy;
    }

    /// Returns a typed slice over the array data, failing if the array is not contiguous.
    pub fn asSlice(self: Array, comptime T: type) ShapeError![]T {
        if (!self.isContiguous()) return ShapeError.IncompatibleShapes;
        std.debug.assert(DType.fromType(T) == self.dtype);
        const ptr: [*]T = @ptrCast(@alignCast(self.data_ptr));
        return ptr[0..self.elementCount()];
    }

    /// Returns an immutable typed slice over the array data, failing if not contiguous.
    pub fn asConstSlice(self: Array, comptime T: type) ShapeError![]const T {
        if (!self.isContiguous()) return ShapeError.IncompatibleShapes;
        std.debug.assert(DType.fromType(T) == self.dtype);
        const ptr: [*]const T = @ptrCast(@alignCast(self.data_ptr));
        return ptr[0..self.elementCount()];
    }

    /// Computes the linear element offset for a multidimensional coordinate index.
    pub fn elementOffset(self: Array, indices: []const usize) IndexError!isize {
        if (indices.len != self.ndim) return IndexError.RankMismatch;
        var offset: isize = 0;
        for (indices, 0..) |idx, dim| {
            if (idx >= self.shape_dims[dim]) return IndexError.IndexOutOfBounds;
            offset += @as(isize, @intCast(idx)) * self.stride_vals[dim];
        }
        return offset;
    }

    /// Gets a scalar value of type T at the specified multidimensional indices.
    pub fn get(self: Array, comptime T: type, indices: []const usize) IndexError!T {
        std.debug.assert(DType.fromType(T) == self.dtype);
        const offset = try self.elementOffset(indices);
        const ptr: [*]const T = @ptrCast(@alignCast(self.data_ptr));
        const final_ptr = if (offset >= 0)
            ptr + @as(usize, @intCast(offset))
        else
            ptr - @as(usize, @intCast(-offset));
        return final_ptr[0];
    }

    /// Alias for `get`.
    pub const getItem = get;

    /// Convenience helper to read a scalar element from a 0D or 1D array as type T.
    /// If no index is provided, reads index 0 / scalar value.
    pub fn item(self: Array, comptime T: type) IndexError!T {
        if (self.ndim == 0) {
            return self.get(T, &.{});
        } else if (self.ndim == 1) {
            return self.get(T, &.{0});
        } else {
            return self.get(T, &([_]usize{0} ** 8)[0..self.ndim]);
        }
    }

    /// Sets a scalar value of type T at the specified multidimensional indices.
    pub fn set(self: Array, comptime T: type, indices: []const usize, value: T) IndexError!void {
        std.debug.assert(DType.fromType(T) == self.dtype);
        const offset = try self.elementOffset(indices);
        const ptr: [*]T = @ptrCast(@alignCast(self.data_ptr));
        const final_ptr = if (offset >= 0)
            ptr + @as(usize, @intCast(offset))
        else
            ptr - @as(usize, @intCast(-offset));
        final_ptr[0] = value;
    }

    /// Convenience helper to set a scalar element on a 0D or 1D array.
    pub fn setItem(self: Array, comptime T: type, value: T) IndexError!void {
        if (self.ndim == 0) {
            try self.set(T, &.{}, value);
        } else if (self.ndim == 1) {
            try self.set(T, &.{0}, value);
        } else {
            const zero_indices = [_]usize{0} ** 8;
            try self.set(T, zero_indices[0..self.ndim], value);
        }
    }

    /// Reads an element at `indices` and converts it to f64 regardless of internal dtype.
    pub fn getAsFloat(self: Array, indices: []const usize) IndexError!f64 {
        const offset = try self.elementOffset(indices);
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (self.dtype == tag) {
                const SrcT = tag.toType();
                const ptr: [*]const SrcT = @ptrCast(@alignCast(self.data_ptr));
                const final_ptr = if (offset >= 0)
                    ptr + @as(usize, @intCast(offset))
                else
                    ptr - @as(usize, @intCast(-offset));
                const elem = final_ptr[0];
                return switch (@typeInfo(SrcT)) {
                    .float => @floatCast(elem),
                    .int => @floatFromInt(elem),
                    .bool => if (elem) 1.0 else 0.0,
                    .@"struct" => elem.re, // Complex(T) returns real part
                    else => 0.0,
                };
            }
        }
        return 0.0;
    }

    /// Sets an element at `indices` from an f64 value, casting to the array's dtype.
    pub fn setFromFloat(self: Array, indices: []const usize, val: f64) IndexError!void {
        const offset = try self.elementOffset(indices);
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (self.dtype == tag) {
                const DstT = tag.toType();
                const ptr: [*]DstT = @ptrCast(@alignCast(self.data_ptr));
                const final_ptr = if (offset >= 0)
                    ptr + @as(usize, @intCast(offset))
                else
                    ptr - @as(usize, @intCast(-offset));
                final_ptr[0] = switch (@typeInfo(DstT)) {
                    .float => @floatCast(val),
                    .int => @intFromFloat(val),
                    .bool => val != 0.0,
                    .@"struct" => .{ .re = @floatCast(val), .im = 0.0 },
                    else => 0,
                };
                return;
            }
        }
    }

    /// Reads an element at `indices` and converts it to i64 regardless of internal dtype.
    pub fn getAsInt(self: Array, indices: []const usize) IndexError!i64 {
        const offset = try self.elementOffset(indices);
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (self.dtype == tag) {
                const SrcT = tag.toType();
                const ptr: [*]const SrcT = @ptrCast(@alignCast(self.data_ptr));
                const final_ptr = if (offset >= 0)
                    ptr + @as(usize, @intCast(offset))
                else
                    ptr - @as(usize, @intCast(-offset));
                const elem = final_ptr[0];
                return switch (@typeInfo(SrcT)) {
                    .float => @intFromFloat(elem),
                    .int => @intCast(elem),
                    .bool => if (elem) 1 else 0,
                    .@"struct" => @intFromFloat(elem.re),
                    else => 0,
                };
            }
        }
        return 0;
    }

    /// Sets an element at `indices` from an i64 value, casting to the array's dtype.
    pub fn setFromInt(self: Array, indices: []const usize, val: i64) IndexError!void {
        const offset = try self.elementOffset(indices);
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (self.dtype == tag) {
                const DstT = tag.toType();
                const ptr: [*]DstT = @ptrCast(@alignCast(self.data_ptr));
                const final_ptr = if (offset >= 0)
                    ptr + @as(usize, @intCast(offset))
                else
                    ptr - @as(usize, @intCast(-offset));
                final_ptr[0] = switch (@typeInfo(DstT)) {
                    .float => @floatFromInt(val),
                    .int => @intCast(val),
                    .bool => val != 0,
                    .@"struct" => .{ .re = @floatFromInt(val), .im = 0.0 },
                    else => 0,
                };
                return;
            }
        }
    }

    /// Read scalar item directly as f64 without requiring client comptime type.
    pub fn itemAsFloat(self: Array) IndexError!f64 {
        if (self.ndim == 0) return self.getAsFloat(&.{});
        if (self.ndim == 1) return self.getAsFloat(&.{0});
        const zeros_buf = [_]usize{0} ** 8;
        return self.getAsFloat(zeros_buf[0..self.ndim]);
    }

    /// Read scalar item directly as i64 without requiring client comptime type.
    pub fn itemAsInt(self: Array) IndexError!i64 {
        if (self.ndim == 0) return self.getAsInt(&.{});
        if (self.ndim == 1) return self.getAsInt(&.{0});
        const zeros_buf = [_]usize{0} ** 8;
        return self.getAsInt(zeros_buf[0..self.ndim]);
    }

    /// Takes elements from an array along an axis according to indices.
    pub fn take(self: Array, indices_arr: Array, options: struct { axis: ?isize = null }) (ShapeError || IndexError || std.mem.Allocator.Error)!Array {
        if (options.axis) |ax| {
            const s = self.shape();
            const norm_ax = try s.normalizeAxis(ax);
            const axis_len = s.dims[norm_ax];
            const n_indices = indices_arr.elementCount();

            var new_shape: [MAX_RANK]usize = undefined;
            var out_ndim: u8 = 0;
            for (0..s.ndim) |d| {
                if (d == norm_ax) {
                    new_shape[out_ndim] = n_indices;
                } else {
                    new_shape[out_ndim] = s.dims[d];
                }
                out_ndim += 1;
            }

            var out = try empty(self.allocator, .{ .shape = new_shape[0..out_ndim], .dtype = self.dtype });
            errdefer out.deinit();

            var it = @import("iterator.zig").NdIterator.init(out.shape(), out.strides());
            while (it.next()) |iter_item| {
                var src_indices: [MAX_RANK]usize = undefined;
                for (0..out_ndim) |d| {
                    if (d == norm_ax) {
                        const out_ax_idx = iter_item.indices[d];
                        // Get flat index from indices_arr
                        var idx_it = @import("iterator.zig").NdIterator.init(indices_arr.shape(), indices_arr.strides());
                        var cur_pos: usize = 0;
                        var raw_idx: i64 = 0;
                        while (idx_it.next()) |idx_item| {
                            if (cur_pos == out_ax_idx) {
                                raw_idx = try indices_arr.getAsInt(idx_item.indices[0..indices_arr.ndim]);
                                break;
                            }
                            cur_pos += 1;
                        }
                        const norm_idx: usize = if (raw_idx < 0)
                            @intCast(@as(isize, @intCast(axis_len)) + raw_idx)
                        else
                            @intCast(raw_idx);
                        if (norm_idx >= axis_len) return IndexError.IndexOutOfBounds;
                        src_indices[d] = norm_idx;
                    } else {
                        src_indices[d] = iter_item.indices[d];
                    }
                }
                const val = try self.getAsFloat(src_indices[0..out_ndim]);
                try out.setFromFloat(iter_item.indices[0..out_ndim], val);
            }
            return out;
        } else {
            const total = self.elementCount();
            var out = try empty(self.allocator, .{ .shape = indices_arr.shapeSlice(), .dtype = self.dtype });
            errdefer out.deinit();

            var out_it = @import("iterator.zig").NdIterator.init(out.shape(), out.strides());
            var idx_it = @import("iterator.zig").NdIterator.init(indices_arr.shape(), indices_arr.strides());

            while (out_it.next()) |out_item| {
                const idx_item = idx_it.next() orelse break;
                const raw_idx = try indices_arr.getAsInt(idx_item.indices[0..indices_arr.ndim]);
                const norm_idx: usize = if (raw_idx < 0)
                    @intCast(@as(isize, @intCast(total)) + raw_idx)
                else
                    @intCast(raw_idx);
                if (norm_idx >= total) return IndexError.IndexOutOfBounds;

                // Find norm_idx in self
                var self_it = @import("iterator.zig").NdIterator.init(self.shape(), self.strides());
                var cur_p: usize = 0;
                var found_val: f64 = 0.0;
                while (self_it.next()) |s_item| {
                    if (cur_p == norm_idx) {
                        found_val = try self.getAsFloat(s_item.indices[0..self.ndim]);
                        break;
                    }
                    cur_p += 1;
                }
                try out.setFromFloat(out_item.indices[0..out.ndim], found_val);
            }
            return out;
        }
    }

    /// Replaces specified elements of an array with given values.
    pub fn put(self: *Array, indices_arr: Array, values_arr: Array) (IndexError || ShapeError || std.mem.Allocator.Error)!void {
        const total = self.elementCount();
        const n_idx = indices_arr.elementCount();
        const n_vals = values_arr.elementCount();
        if (n_vals == 0 or n_idx == 0) return;

        var idx_it = @import("iterator.zig").NdIterator.init(indices_arr.shape(), indices_arr.strides());
        var count: usize = 0;

        while (idx_it.next()) |idx_item| {
            const raw_idx = try indices_arr.getAsInt(idx_item.indices[0..indices_arr.ndim]);
            const norm_idx: usize = if (raw_idx < 0)
                @intCast(@as(isize, @intCast(total)) + raw_idx)
            else
                @intCast(raw_idx);
            if (norm_idx >= total) return IndexError.IndexOutOfBounds;

            const val_target = count % n_vals;
            var val_it = @import("iterator.zig").NdIterator.init(values_arr.shape(), values_arr.strides());
            var val_pos: usize = 0;
            var val: f64 = 0.0;
            while (val_it.next()) |v_item| {
                if (val_pos == val_target) {
                    val = try values_arr.getAsFloat(v_item.indices[0..values_arr.ndim]);
                    break;
                }
                val_pos += 1;
            }

            var self_it = @import("iterator.zig").NdIterator.init(self.shape(), self.strides());
            var cur_p: usize = 0;
            while (self_it.next()) |s_item| {
                if (cur_p == norm_idx) {
                    try self.setFromFloat(s_item.indices[0..self.ndim], val);
                    break;
                }
                cur_p += 1;
            }
            count += 1;
        }
    }

    /// Fills the array in-place with a constant scalar value.
    pub fn fill(self: *Array, comptime T: type, val: T) void {
        inline for (std.meta.fields(DType)) |field| {
            const tag: DType = @enumFromInt(field.value);
            if (self.dtype == tag) {
                const DstT = tag.toType();
                const cast_val: DstT = switch (@typeInfo(DstT)) {
                    .int => switch (@typeInfo(T)) {
                        .int, .comptime_int => @as(DstT, @intCast(val)),
                        .float, .comptime_float => @as(DstT, @intFromFloat(val)),
                        .bool => @as(DstT, if (val) 1 else 0),
                        else => 0,
                    },
                    .float => switch (@typeInfo(T)) {
                        .int, .comptime_int => @as(DstT, @floatFromInt(val)),
                        .float, .comptime_float => @as(DstT, @floatCast(val)),
                        .bool => @as(DstT, if (val) 1.0 else 0.0),
                        else => 0.0,
                    },
                    .bool => switch (@typeInfo(T)) {
                        .bool => val,
                        .int, .comptime_int => val != 0,
                        .float, .comptime_float => val != 0.0,
                        else => false,
                    },
                    .@"struct" => switch (@typeInfo(T)) {
                        .float, .comptime_float => .{ .re = @floatCast(val), .im = 0.0 },
                        .int, .comptime_int => .{ .re = @floatFromInt(val), .im = 0.0 },
                        else => .{ .re = 0.0, .im = 0.0 },
                    },
                    else => 0,
                };

                if (self.flags.isCContiguous) {
                    const ptr: [*]DstT = @ptrCast(@alignCast(self.data_ptr));
                    const n = self.elementCount();
                    @memset(ptr[0..n], cast_val);
                } else {
                    var it = @import("iterator.zig").NdIterator.init(self.shape(), self.strides());
                    while (it.next()) |iter_item| {
                        self.set(DstT, iter_item.indices[0..self.ndim], cast_val) catch unreachable;
                    }
                }
                return;
            }
        }
    }
};

/// Creates an array of the given shape and data type filled with zeros.
pub fn zeros(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = try Shape.init(options.shape);
    const elem_count = try s.elementCountChecked();
    const buf = try Buffer.allocZeroed(allocator, options.dtype, elem_count);

    const st = Strides.fromShape(s, options.order);

    var arr = Array{
        .allocator = allocator,
        .data_ptr = buf.bytes.ptr,
        .dtype = options.dtype,
        .ndim = s.ndim,
        .flags = .{
            .ownsData = true,
            .isCContiguous = Strides.isCContiguous(s, st),
            .isFContiguous = Strides.isFContiguous(s, st),
        },
        .root_buffer = buf,
    };

    for (0..s.ndim) |i| {
        arr.shape_dims[i] = s.dims[i];
        arr.stride_vals[i] = st.values[i];
    }

    return arr;
}

/// Creates an uninitialized array of the given shape and data type.
pub fn empty(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = try Shape.init(options.shape);
    const elem_count = try s.elementCountChecked();
    const buf = try Buffer.alloc(allocator, options.dtype, elem_count);

    const st = Strides.fromShape(s, options.order);

    var arr = Array{
        .allocator = allocator,
        .data_ptr = buf.bytes.ptr,
        .dtype = options.dtype,
        .ndim = s.ndim,
        .flags = .{
            .ownsData = true,
            .isCContiguous = Strides.isCContiguous(s, st),
            .isFContiguous = Strides.isFContiguous(s, st),
        },
        .root_buffer = buf,
    };

    for (0..s.ndim) |i| {
        arr.shape_dims[i] = s.dims[i];
        arr.stride_vals[i] = st.values[i];
    }

    return arr;
}

/// Creates an array of the given shape filled with ones.
pub fn ones(
    allocator: std.mem.Allocator,
    options: struct {
        shape: []const usize,
        dtype: DType = .f64,
        order: Order = .c,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    return full(allocator, .{
        .shape = options.shape,
        .value = 1.0,
        .dtype = options.dtype,
        .order = options.order,
    });
}

/// Creates an array of the given shape filled with a constant scalar value.
pub fn full(
    allocator: std.mem.Allocator,
    options: anytype,
) (ShapeError || std.mem.Allocator.Error)!Array {
    const ValType = @TypeOf(options.value);
    const resolved_dtype: DType = blk: {
        if (@hasField(@TypeOf(options), "dtype")) {
            if (@typeInfo(@TypeOf(options.dtype)) == .optional) {
                if (options.dtype) |dt| break :blk dt;
            } else {
                break :blk options.dtype;
            }
        }
        break :blk DType.fromType(ValType);
    };
    const order = if (@hasField(@TypeOf(options), "order")) options.order else Order.c;

    var arr = try empty(allocator, .{
        .shape = options.shape,
        .dtype = resolved_dtype,
        .order = order,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();
            const slice = arr.asSlice(T) catch unreachable;
            const cast_val: T = switch (@typeInfo(T)) {
                .int => switch (@typeInfo(ValType)) {
                    .int, .comptime_int => @as(T, @intCast(options.value)),
                    .float, .comptime_float => @as(T, @intFromFloat(options.value)),
                    .bool => @as(T, if (options.value) 1 else 0),
                    else => @as(T, 0),
                },
                .float => switch (@typeInfo(ValType)) {
                    .int, .comptime_int => @as(T, @floatFromInt(options.value)),
                    .float, .comptime_float => @as(T, @floatCast(options.value)),
                    .bool => @as(T, if (options.value) 1.0 else 0.0),
                    else => @as(T, 0.0),
                },
                .bool => if (options.value != 0) true else false,
                .@"struct" => switch (@typeInfo(ValType)) {
                    .@"struct" => .{ .re = @floatCast(options.value.re), .im = @floatCast(options.value.im) },
                    .int, .comptime_int => .{ .re = @floatFromInt(options.value), .im = 0.0 },
                    .float, .comptime_float => .{ .re = @floatCast(options.value), .im = 0.0 },
                    else => .{ .re = 0.0, .im = 0.0 },
                },
                else => unreachable,
            };
            @memset(slice, cast_val);
            return arr;
        }
    }

    return arr;
}

/// Creates a 1D array with values evenly spaced within a half-open interval [start, stop).
pub fn arange(
    allocator: std.mem.Allocator,
    options: anytype,
) (ShapeError || std.mem.Allocator.Error)!Array {
    const OptType = @TypeOf(options);
    const StopType = @TypeOf(options.stop);

    const has_start = comptime blk: {
        if (@hasField(OptType, "start")) {
            if (@typeInfo(@TypeOf(options.start)) == .optional) {
                break :blk options.start != null;
            }
            break :blk true;
        }
        break :blk false;
    };

    const has_step = comptime blk: {
        if (@hasField(OptType, "step")) {
            if (@typeInfo(@TypeOf(options.step)) == .optional) {
                break :blk options.step != null;
            }
            break :blk true;
        }
        break :blk false;
    };

    const resolved_dtype: DType = blk: {
        if (@hasField(OptType, "dtype")) {
            if (@typeInfo(@TypeOf(options.dtype)) == .optional) {
                if (options.dtype) |dt| break :blk dt;
            } else {
                break :blk options.dtype;
            }
        }
        var d = DType.fromType(StopType);
        if (has_start) d = DType.promote(d, DType.fromType(@TypeOf(options.start)));
        if (has_step) d = DType.promote(d, DType.fromType(@TypeOf(options.step)));
        break :blk d;
    };

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (resolved_dtype == tag) {
            const T = tag.toType();

            const start_val: T = if (has_start) switch (@typeInfo(T)) {
                .int => @as(T, @intCast(options.start)),
                .float => @as(T, @floatCast(options.start)),
                else => 0,
            } else 0;

            const stop_val: T = switch (@typeInfo(T)) {
                .int => @as(T, @intCast(options.stop)),
                .float => @as(T, @floatCast(options.stop)),
                else => 0,
            };

            const step_val: T = if (has_step) switch (@typeInfo(T)) {
                .int => @as(T, @intCast(options.step)),
                .float => @as(T, @floatCast(options.step)),
                else => 1,
            } else 1;

            var count: usize = 0;
            switch (@typeInfo(T)) {
                .int => {
                    if (step_val > 0 and stop_val > start_val) {
                        count = @intCast(@divFloor(stop_val - start_val - 1, step_val) + 1);
                    } else if (step_val < 0 and start_val > stop_val) {
                        count = @intCast(@divFloor(start_val - stop_val - 1, -step_val) + 1);
                    }
                },
                .float => {
                    if (step_val > 0 and stop_val > start_val) {
                        count = @intFromFloat(@ceil((stop_val - start_val) / step_val));
                    } else if (step_val < 0 and start_val > stop_val) {
                        count = @intFromFloat(@ceil((start_val - stop_val) / -step_val));
                    }
                },
                else => {},
            }

            var arr = try empty(allocator, .{
                .shape = &.{count},
                .dtype = resolved_dtype,
            });

            const slice = arr.asSlice(T) catch unreachable;
            var curr = start_val;
            for (slice) |*elem| {
                elem.* = curr;
                curr += step_val;
            }

            return arr;
        }
    }

    return ShapeError.InvalidDimension;
}

/// Creates a 1D array of `num` evenly spaced numbers over the interval [start, stop].
pub fn linspace(
    allocator: std.mem.Allocator,
    options: anytype,
) (ShapeError || std.mem.Allocator.Error)!Array {
    const num_val: usize = if (@hasField(@TypeOf(options), "num")) options.num else 50;
    const endpoint: bool = if (@hasField(@TypeOf(options), "endpoint")) options.endpoint else true;
    const resolved_dtype: DType = if (@hasField(@TypeOf(options), "dtype")) options.dtype else .f64;

    var arr = try empty(allocator, .{
        .shape = &.{num_val},
        .dtype = resolved_dtype,
    });

    if (num_val == 0) return arr;

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();
            const slice = arr.asSlice(T) catch unreachable;

            const start_f: f64 = switch (@typeInfo(@TypeOf(options.start))) {
                .int, .comptime_int => @floatFromInt(options.start),
                .float, .comptime_float => @floatCast(options.start),
                else => 0.0,
            };
            const stop_f: f64 = switch (@typeInfo(@TypeOf(options.stop))) {
                .int, .comptime_int => @floatFromInt(options.stop),
                .float, .comptime_float => @floatCast(options.stop),
                else => 0.0,
            };

            if (num_val == 1) {
                slice[0] = switch (@typeInfo(T)) {
                    .float => @floatCast(start_f),
                    .int => @intFromFloat(start_f),
                    .bool => start_f != 0.0,
                    else => unreachable,
                };
                return arr;
            }

            const step = if (endpoint)
                (stop_f - start_f) / @as(f64, @floatFromInt(num_val - 1))
            else
                (stop_f - start_f) / @as(f64, @floatFromInt(num_val));

            for (slice, 0..) |*elem, i| {
                const val = start_f + @as(f64, @floatFromInt(i)) * step;
                elem.* = switch (@typeInfo(T)) {
                    .float => @floatCast(val),
                    .int => @intFromFloat(val),
                    .bool => val != 0.0,
                    else => unreachable,
                };
            }

            return arr;
        }
    }

    return arr;
}

/// Creates a 1D array with numbers spaced evenly on a log scale.
pub fn logspace(
    allocator: std.mem.Allocator,
    options: anytype,
) (ShapeError || IndexError || std.mem.Allocator.Error)!Array {
    const base: f64 = if (@hasField(@TypeOf(options), "base")) @floatCast(options.base) else 10.0;
    const num_val: usize = if (@hasField(@TypeOf(options), "num")) options.num else 50;
    const endpoint: bool = if (@hasField(@TypeOf(options), "endpoint")) options.endpoint else true;
    const dtype: DType = if (@hasField(@TypeOf(options), "dtype")) options.dtype else .f64;

    var lin = try linspace(allocator, .{
        .start = options.start,
        .stop = options.stop,
        .num = num_val,
        .endpoint = endpoint,
        .dtype = dtype,
    });
    defer lin.deinit();

    var out = try empty(allocator, .{ .shape = &.{num_val}, .dtype = dtype });
    errdefer out.deinit();

    for (0..num_val) |i| {
        const exponent = lin.getAsFloat(&.{i}) catch unreachable;
        const val = std.math.pow(f64, base, exponent);
        try out.setFromFloat(&.{i}, val);
    }
    return out;
}

/// Creates a 1D array with numbers spaced evenly on a geometric progression.
pub fn geomspace(
    allocator: std.mem.Allocator,
    options: anytype,
) (ShapeError || IndexError || std.mem.Allocator.Error)!Array {
    const num_val: usize = if (@hasField(@TypeOf(options), "num")) options.num else 50;
    const endpoint: bool = if (@hasField(@TypeOf(options), "endpoint")) options.endpoint else true;
    const dtype: DType = if (@hasField(@TypeOf(options), "dtype")) options.dtype else .f64;

    const start_f: f64 = switch (@typeInfo(@TypeOf(options.start))) {
        .int, .comptime_int => @floatFromInt(options.start),
        .float, .comptime_float => @floatCast(options.start),
        else => 1.0,
    };
    const stop_f: f64 = switch (@typeInfo(@TypeOf(options.stop))) {
        .int, .comptime_int => @floatFromInt(options.stop),
        .float, .comptime_float => @floatCast(options.stop),
        else => 1.0,
    };

    if (start_f <= 0.0 or stop_f <= 0.0) return ShapeError.InvalidDimension;

    const log_start = @log10(start_f);
    const log_stop = @log10(stop_f);

    return logspace(allocator, .{
        .start = log_start,
        .stop = log_stop,
        .num = num_val,
        .endpoint = endpoint,
        .base = 10.0,
        .dtype = dtype,
    });
}

/// Upper triangle of an array. Returns a copy of a matrix with elements below the k-th diagonal zeroed.
pub fn triu(
    m: Array,
    options: struct { k: isize = 0 },
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (m.ndim < 2) return ShapeError.InvalidDimension;
    var out = try m.clone();
    errdefer out.deinit();

    const s = m.shape();
    const rows = s.dims[s.ndim - 2];
    const cols = s.dims[s.ndim - 1];

    var it = @import("iterator.zig").NdIterator.init(s, m.strides());
    while (it.next()) |item| {
        const r = item.indices[s.ndim - 2];
        const c = item.indices[s.ndim - 1];
        _ = cols;
        _ = rows;
        const c_isize: isize = @intCast(c);
        const r_isize: isize = @intCast(r);
        if (c_isize < r_isize + options.k) {
            out.setFromFloat(item.indices[0..s.ndim], 0.0) catch unreachable;
        }
    }

    return out;
}

/// Lower triangle of an array. Returns a copy of a matrix with elements above the k-th diagonal zeroed.
pub fn tril(
    m: Array,
    options: struct { k: isize = 0 },
) (ShapeError || std.mem.Allocator.Error)!Array {
    if (m.ndim < 2) return ShapeError.InvalidDimension;
    var out = try m.clone();
    errdefer out.deinit();

    const s = m.shape();
    const rows = s.dims[s.ndim - 2];
    const cols = s.dims[s.ndim - 1];

    var it = @import("iterator.zig").NdIterator.init(s, m.strides());
    while (it.next()) |item| {
        const r = item.indices[s.ndim - 2];
        const c = item.indices[s.ndim - 1];
        _ = cols;
        _ = rows;
        const c_isize: isize = @intCast(c);
        const r_isize: isize = @intCast(r);
        if (c_isize > r_isize + options.k) {
            out.setFromFloat(item.indices[0..s.ndim], 0.0) catch unreachable;
        }
    }

    return out;
}

/// Extracts a diagonal or constructs a diagonal 2D matrix.
pub fn diag(allocator: std.mem.Allocator, v: Array, options: struct { k: isize = 0 }) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const k = options.k;
    if (v.ndim == 1) {
        const n = v.shape_dims[0];
        const abs_k: usize = @intCast(@abs(k));
        const dim = n + abs_k;
        var out = try zeros(allocator, .{ .shape = &.{ dim, dim }, .dtype = v.dtype });
        errdefer out.deinit();

        for (0..n) |i| {
            const r: usize = if (k >= 0) i else i + abs_k;
            const c: usize = if (k >= 0) i + abs_k else i;
            const val = try v.getAsFloat(&.{i});
            try out.setFromFloat(&.{ r, c }, val);
        }
        return out;
    } else if (v.ndim == 2) {
        const rows = v.shape_dims[0];
        const cols = v.shape_dims[1];
        var count: usize = 0;
        for (0..rows) |r| {
            const c_isize = @as(isize, @intCast(r)) + k;
            if (c_isize >= 0 and c_isize < cols) count += 1;
        }
        var out = try empty(allocator, .{ .shape = &.{count}, .dtype = v.dtype });
        errdefer out.deinit();

        var idx: usize = 0;
        for (0..rows) |r| {
            const c_isize = @as(isize, @intCast(r)) + k;
            if (c_isize >= 0 and c_isize < cols) {
                const val = try v.getAsFloat(&.{ r, @intCast(c_isize) });
                try out.setFromFloat(&.{idx}, val);
                idx += 1;
            }
        }
        return out;
    } else {
        return ShapeError.InvalidDimension;
    }
}

/// Creates a 2D identity matrix with ones on the main diagonal.
pub fn identity(
    allocator: std.mem.Allocator,
    options: struct {
        n: usize,
        dtype: DType = .f64,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    return eye(allocator, .{
        .n = options.n,
        .m = options.n,
        .k = 0,
        .dtype = options.dtype,
    });
}

/// Creates a 2D array with ones on the k-th diagonal and zeros elsewhere.
pub fn eye(
    allocator: std.mem.Allocator,
    options: struct {
        n: usize,
        m: ?usize = null,
        k: isize = 0,
        dtype: DType = .f64,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const cols = options.m orelse options.n;
    var arr = try zeros(allocator, .{
        .shape = &.{ options.n, cols },
        .dtype = options.dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (arr.dtype == tag) {
            const T = tag.toType();
            const one_val: T = switch (@typeInfo(T)) {
                .int => 1,
                .float => 1.0,
                .bool => true,
                .@"struct" => .{ .re = 1.0, .im = 0.0 },
                else => 1,
            };

            for (0..options.n) |r| {
                const c_isize = @as(isize, @intCast(r)) + options.k;
                if (c_isize >= 0 and c_isize < cols) {
                    const c: usize = @intCast(c_isize);
                    arr.set(T, &.{ r, c }, one_val) catch unreachable;
                }
            }

            return arr;
        }
    }

    return arr;
}

/// Creates an array by copying data from a native Zig slice.
pub fn fromSlice(
    allocator: std.mem.Allocator,
    comptime T: type,
    options: struct {
        data: []const T,
        shape: ?[]const usize = null,
        order: Order = .c,
    },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const dtype = DType.fromType(T);
    const target_shape = if (options.shape) |shp|
        try Shape.init(shp)
    else
        Shape.vector(options.data.len);

    if (target_shape.elementCount() != options.data.len) {
        return ShapeError.ReshapeMismatch;
    }

    const buf = try Buffer.fromSlice(allocator, T, options.data);
    const st = Strides.fromShape(target_shape, options.order);

    var arr = Array{
        .allocator = allocator,
        .data_ptr = buf.bytes.ptr,
        .dtype = dtype,
        .ndim = target_shape.ndim,
        .flags = .{
            .ownsData = true,
            .isCContiguous = Strides.isCContiguous(target_shape, st),
            .isFContiguous = Strides.isFContiguous(target_shape, st),
        },
        .root_buffer = buf,
    };

    for (0..target_shape.ndim) |i| {
        arr.shape_dims[i] = target_shape.dims[i];
        arr.stride_vals[i] = st.values[i];
    }

    return arr;
}

test "zeros and ones creation" {
    const allocator = std.testing.allocator;

    var z = try zeros(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f32 });
    defer z.deinit();

    try std.testing.expectEqual(@as(usize, 6), z.elementCount());
    try std.testing.expect(z.isContiguous());
    const z_slice = try z.asSlice(f32);
    for (z_slice) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }

    var o = try ones(allocator, .{ .shape = &.{4}, .dtype = .i32 });
    defer o.deinit();

    const o_slice = try o.asSlice(i32);
    for (o_slice) |v| {
        try std.testing.expectEqual(@as(i32, 1), v);
    }
}

test "arange and linspace" {
    const allocator = std.testing.allocator;

    var a = try arange(allocator, .{ .start = 0, .stop = 5, .dtype = .f32 });
    defer a.deinit();

    try std.testing.expectEqual(@as(usize, 5), a.elementCount());
    const a_slice = try a.asSlice(f32);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 3.0, 4.0 }, a_slice);

    var lin = try linspace(allocator, .{ .start = 0.0, .stop = 1.0, .num = 5, .dtype = .f64 });
    defer lin.deinit();

    const lin_slice = try lin.asSlice(f64);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), lin_slice[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), lin_slice[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), lin_slice[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), lin_slice[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), lin_slice[4], 1e-6);
}

test "eye and identity matrices" {
    const allocator = std.testing.allocator;

    var id = try identity(allocator, .{ .n = 3, .dtype = .f32 });
    defer id.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), try id.get(f32, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 0.0), try id.get(f32, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try id.get(f32, &.{ 1, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try id.get(f32, &.{ 2, 2 }));

    var e = try eye(allocator, .{ .n = 3, .m = 3, .k = 1, .dtype = .f32 });
    defer e.deinit();

    try std.testing.expectEqual(@as(f32, 1.0), try e.get(f32, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 1.0), try e.get(f32, &.{ 1, 2 }));
    try std.testing.expectEqual(@as(f32, 0.0), try e.get(f32, &.{ 0, 0 }));
}

test "fromSlice and view semantics" {
    const allocator = std.testing.allocator;

    const data = [_]f64{ 10, 20, 30, 40 };
    var arr = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 2 } });
    defer arr.deinit();

    try std.testing.expectEqual(@as(f64, 10.0), try arr.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 40.0), try arr.get(f64, &.{ 1, 1 }));

    var v = arr.view();
    defer v.deinit(); // Safe no-op because owns_data is false

    try v.set(f64, &.{ 0, 0 }, 99.0);
    try std.testing.expectEqual(@as(f64, 99.0), try arr.get(f64, &.{ 0, 0 }));
}

test "scalar item and type-agnostic get/set accessors" {
    const allocator = std.testing.allocator;

    const data = [_]i32{ 7, 14, 21, 28 };
    var arr = try fromSlice(allocator, i32, .{ .data = &data, .shape = &.{ 2, 2 } });
    defer arr.deinit();

    // getAsFloat on i32 array
    try std.testing.expectEqual(@as(f64, 7.0), try arr.getAsFloat(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 28.0), try arr.getAsFloat(&.{ 1, 1 }));

    // setFromFloat on i32 array
    try arr.setFromFloat(&.{ 0, 1 }, 42.0);
    try std.testing.expectEqual(@as(i64, 42), try arr.getAsInt(&.{ 0, 1 }));

    // 0D scalar array item access
    const scalar_data = [_]f32{3.1415};
    var sc = try fromSlice(allocator, f32, .{ .data = &scalar_data, .shape = &.{} });
    defer sc.deinit();

    try std.testing.expectApproxEqAbs(@as(f64, 3.1415), try sc.itemAsFloat(), 1e-4);
    try std.testing.expectEqual(@as(i64, 3), try sc.itemAsInt());
}

test "geomspace, triu, and tril" {
    const allocator = std.testing.allocator;

    // geomspace 1.0 to 1000.0 (4 points: 1, 10, 100, 1000)
    var g = try geomspace(allocator, .{ .start = 1.0, .stop = 1000.0, .num = 4, .dtype = .f64 });
    defer g.deinit();

    try std.testing.expectEqual(@as(usize, 4), g.elementCount());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try g.get(f64, &.{0}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), try g.get(f64, &.{1}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), try g.get(f64, &.{2}), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), try g.get(f64, &.{3}), 1e-5);

    // triu and tril on 3x3
    const mat_data = [_]f64{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    };
    var mat = try fromSlice(allocator, f64, .{ .data = &mat_data, .shape = &.{ 3, 3 } });
    defer mat.deinit();

    var u = try triu(mat, .{});
    defer u.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), try u.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 2.0), try u.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 0.0), try u.get(f64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 0.0), try u.get(f64, &.{ 2, 0 }));

    var l = try tril(mat, .{});
    defer l.deinit();
    try std.testing.expectEqual(@as(f64, 1.0), try l.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 0.0), try l.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 4.0), try l.get(f64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 7.0), try l.get(f64, &.{ 2, 0 }));
}

test "take, put, and fill methods" {
    const allocator = std.testing.allocator;

    const v_data = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &v_data, .shape = &.{5} });
    defer arr.deinit();

    // take flat indices: 0, 3, 1
    const idx_data = [_]i64{ 0, 3, 1 };
    var idx_arr = try fromSlice(allocator, i64, .{ .data = &idx_data, .shape = &.{3} });
    defer idx_arr.deinit();

    var taken = try arr.take(idx_arr, .{});
    defer taken.deinit();
    try std.testing.expectEqualSlices(usize, &.{3}, taken.shapeSlice());
    try std.testing.expectEqual(@as(f64, 10.0), try taken.get(f64, &.{0}));
    try std.testing.expectEqual(@as(f64, 40.0), try taken.get(f64, &.{1}));
    try std.testing.expectEqual(@as(f64, 20.0), try taken.get(f64, &.{2}));

    // put values
    const put_idx = [_]i64{ 1, 3 };
    var put_idx_arr = try fromSlice(allocator, i64, .{ .data = &put_idx, .shape = &.{2} });
    defer put_idx_arr.deinit();
    const put_val = [_]f64{ 99.0, 77.0 };
    var put_val_arr = try fromSlice(allocator, f64, .{ .data = &put_val, .shape = &.{2} });
    defer put_val_arr.deinit();

    try arr.put(put_idx_arr, put_val_arr);
    try std.testing.expectEqual(@as(f64, 99.0), try arr.get(f64, &.{1}));
    try std.testing.expectEqual(@as(f64, 77.0), try arr.get(f64, &.{3}));

    // fill in-place
    arr.fill(f64, 42.0);
    try std.testing.expectEqual(@as(f64, 42.0), try arr.get(f64, &.{0}));
    try std.testing.expectEqual(@as(f64, 42.0), try arr.get(f64, &.{4}));

    // getItem alias check
    try std.testing.expectEqual(@as(f64, 42.0), try arr.getItem(f64, &.{0}));
    try arr.setItem(f64, 100.0);
    try std.testing.expectEqual(@as(f64, 100.0), try arr.getItem(f64, &.{0}));
}
