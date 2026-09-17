//! NZIG Version 1.0 — Native array serialization for num.zig.
//!
//! NZIG is the native binary array serialization format for num.zig.
//! Features a deterministic 128-byte 64-byte-aligned header with explicit
//! little-endian metadata, stable dtype mappings, logical view streaming,
//! and complete malformed/truncated data rejection.

const std = @import("std");
const builtin = @import("builtin");
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;
const fromSlice = @import("../core/array.zig").fromSlice;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const NdIterator = @import("../core/iterator.zig").NdIterator;
const IoError = @import("../core/error.zig").IoError;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

/// NZIG file signature: 8 bytes.
pub const MAGIC: [8]u8 = .{ 0x89, 'N', 'Z', 'I', 'G', '1', '0', '\n' };

/// NZIG specification version 1.0.
pub const VERSION_MAJOR: u8 = 1;
pub const VERSION_MINOR: u8 = 0;

/// Header size: exactly 128 bytes, 64-byte aligned for portable zero-overhead I/O.
pub const HEADER_SIZE: usize = 128;

/// Stable NZIG DType identifiers.
pub fn dtypeToId(dtype: DType) u16 {
    return switch (dtype) {
        .bool => 0,
        .i8 => 1,
        .i16 => 2,
        .i32 => 3,
        .i64 => 4,
        .u8 => 5,
        .u16 => 6,
        .u32 => 7,
        .u64 => 8,
        .f16 => 9,
        .f32 => 10,
        .f64 => 11,
        .c64 => 12,
        .c128 => 13,
    };
}

/// Resolves stable NZIG DType identifier to library DType.
pub fn idToDType(id: u16) ?DType {
    return switch (id) {
        0 => .bool,
        1 => .i8,
        2 => .i16,
        3 => .i32,
        4 => .i64,
        5 => .u8,
        6 => .u16,
        7 => .u32,
        8 => .u64,
        9 => .f16,
        10 => .f32,
        11 => .f64,
        12 => .c64,
        13 => .c128,
        else => null,
    };
}

/// Swaps byte order in-place for multi-byte array payloads.
fn swapEndianness(slice: []u8, elem_size: usize) void {
    if (elem_size <= 1) return;
    var i: usize = 0;
    while (i < slice.len) : (i += elem_size) {
        std.mem.reverse(u8, slice[i .. i + elem_size]);
    }
}

/// Serializes an array into a writer in the NZIG v1.0 binary format.
///
/// Handles contiguous arrays with zero-copy stream writes.
/// Non-contiguous views (strided slices, transposes) are serialized
/// by streaming their logical C-order elements.
pub fn writeToStream(
    arr: Array,
    writer: anytype,
) !void {
    var header = [_]u8{0} ** HEADER_SIZE;

    // 0..7: NZIG file signature
    @memcpy(header[0..8], &MAGIC);

    // 8..9: Version
    header[8] = VERSION_MAJOR;
    header[9] = VERSION_MINOR;

    // 10..11: Format flags (little-endian u16)
    std.mem.writeInt(u16, header[10..12], 0, .little);

    // 12..13: Stable DType ID (little-endian u16)
    const dtype_id = dtypeToId(arr.dtype);
    std.mem.writeInt(u16, header[12..14], dtype_id, .little);

    // 14: Byte order (0 = little-endian, 1 = big-endian)
    const host_is_big = (builtin.cpu.arch.endian() == .big);
    header[14] = if (host_is_big) 1 else 0;

    // 15: Memory order (0 = C-order / row-major)
    header[15] = 0;

    // 16: Rank (0..MAX_RANK)
    header[16] = @intCast(arr.ndim);

    // 17..23: Reserved (already 0)

    // 24..31: Payload size in bytes (little-endian u64)
    const elem_size = arr.dtype.sizeOf();
    const elem_count = arr.elementCount();
    const payload_size_usize = std.math.mul(usize, elem_count, elem_size) catch return IoError.InvalidPayloadSize;
    const payload_size_u64: u64 = payload_size_usize;
    std.mem.writeInt(u64, header[24..32], payload_size_u64, .little);

    // 32..95: Dimensions as 8 * u64 (little-endian)
    for (0..arr.ndim) |i| {
        const dim_val: u64 = arr.shape_dims[i];
        std.mem.writeInt(u64, header[32 + i * 8 ..][0..8], dim_val, .little);
    }

    // 96..127: Reserved extension fields (already 0)

    // Write deterministic 128-byte header
    try writer.writeAll(&header);

    // If empty payload (e.g. zero elements), nothing more to stream
    if (payload_size_usize == 0) return;

    // Write payload:
    // If contiguous C-order, stream directly.
    // Otherwise, iterate logical elements in C-order.
    if (arr.isContiguous()) {
        const payload = arr.data_ptr[0..payload_size_usize];
        try writer.writeAll(payload);
    } else {
        var iter = NdIterator.init(arr.shape(), arr.strides());
        var elem_buf: [16]u8 = undefined;

        while (iter.next()) |item| {
            const elem_offset: usize = @intCast(item.offset);
            const src = arr.data_ptr[elem_offset * elem_size .. (elem_offset + 1) * elem_size];
            @memcpy(elem_buf[0..elem_size], src);
            try writer.writeAll(elem_buf[0..elem_size]);
        }
    }
}

/// Deserializes an array from a reader in the NZIG v1.0 binary format.
///
/// Validates magic, versions, flags, dtype, rank, dimensions, and payload sizes.
/// Returns an owned Array allocated with the supplied allocator.
pub fn readFromStream(
    allocator: std.mem.Allocator,
    reader: anytype,
) !Array {
    var header: [HEADER_SIZE]u8 = undefined;

    // Read full 128-byte header
    var header_read: usize = 0;
    while (header_read < HEADER_SIZE) {
        const n = try reader.read(header[header_read..]);
        if (n == 0) return IoError.TruncatedHeader;
        header_read += n;
    }

    // 1. Validate NZIG file signature
    if (!std.mem.eql(u8, header[0..8], &MAGIC)) {
        return IoError.InvalidMagic;
    }

    // 2. Validate version
    const major = header[8];
    const minor = header[9];
    if (major != VERSION_MAJOR) {
        return IoError.UnsupportedVersion;
    }
    if (minor > VERSION_MINOR) {
        return IoError.UnsupportedVersion;
    }

    // 3. Validate flags
    const flags = std.mem.readInt(u16, header[10..12], .little);
    if (flags != 0) {
        return IoError.UnsupportedFlags;
    }

    // 4. Validate DType
    const dtype_id = std.mem.readInt(u16, header[12..14], .little);
    const dtype = idToDType(dtype_id) orelse return IoError.InvalidDType;

    // 5. Validate byte order and memory order
    const file_byte_order = header[14];
    if (file_byte_order > 1) {
        return IoError.InvalidByteOrder;
    }
    const mem_order = header[15];
    if (mem_order != 0) {
        return IoError.InvalidOrder;
    }

    // 6. Validate rank
    const rank = header[16];
    if (rank > MAX_RANK) {
        return IoError.InvalidRank;
    }

    // 7. Validate payload size
    const payload_size_u64 = std.mem.readInt(u64, header[24..32], .little);

    // 8. Validate dimensions and calculate expected payload
    var shape_buf: [MAX_RANK]usize = undefined;
    var calc_elems: usize = 1;

    for (0..rank) |i| {
        const d_u64 = std.mem.readInt(u64, header[32 + i * 8 ..][0..8], .little);
        if (d_u64 > std.math.maxInt(usize)) {
            return IoError.InvalidShape;
        }
        const d: usize = @intCast(d_u64);
        shape_buf[i] = d;

        calc_elems = std.math.mul(usize, calc_elems, d) catch return IoError.InvalidShape;
    }

    const elem_size = dtype.sizeOf();
    const expected_payload_usize = std.math.mul(usize, calc_elems, elem_size) catch return IoError.InvalidPayloadSize;
    const expected_payload_u64: u64 = expected_payload_usize;

    if (payload_size_u64 != expected_payload_u64) {
        return IoError.InvalidPayloadSize;
    }

    // 9. Allocate array
    var arr = try empty(allocator, .{
        .shape = shape_buf[0..rank],
        .dtype = dtype,
    });
    errdefer arr.deinit();

    // 10. Read payload
    if (expected_payload_usize > 0) {
        var payload_read: usize = 0;
        const dest = arr.data_ptr[0..expected_payload_usize];

        while (payload_read < expected_payload_usize) {
            const n = try reader.read(dest[payload_read..]);
            if (n == 0) return IoError.TruncatedPayload;
            payload_read += n;
        }

        // 11. Handle endianness conversion if file byte order != host CPU byte order
        const host_is_big: u8 = if (builtin.cpu.arch.endian() == .big) 1 else 0;
        if (file_byte_order != host_is_big) {
            swapEndianness(dest, elem_size);
        }
    }

    return arr;
}

pub const FileWriter = struct {
    file: std.Io.File,
    io: std.Io,

    pub fn writeAll(self: FileWriter, bytes: []const u8) !void {
        try self.file.writeStreamingAll(self.io, bytes);
    }
};

pub const FileReader = struct {
    file: std.Io.File,
    io: std.Io,
    pos: u64 = 0,

    pub fn read(self: *FileReader, dest: []u8) !usize {
        const amt = try self.file.readPositionalAll(self.io, dest, self.pos);
        self.pos += amt;
        return amt;
    }
};

/// Serializes an array into a `.nzig` file at the specified filesystem path.
pub fn writeFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    arr: Array,
) !void {
    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();

    var file = try std.Io.Dir.cwd().createFile(io, file_path, .{});
    defer file.close(io);

    const fw = FileWriter{ .file = file, .io = io };
    try writeToStream(arr, fw);
}

/// Deserializes an array from a `.nzig` file at the specified filesystem path.
pub fn readFile(
    allocator: std.mem.Allocator,
    file_path: []const u8,
) !Array {
    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();

    var file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
    defer file.close(io);

    var fr = FileReader{ .file = file, .io = io };
    return try readFromStream(allocator, &fr);
}

// ============================================================================
// TESTS
// ============================================================================

test "NZIG round-trip 1D array" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 1.5, -2.25, 3.125, 4.0625 };
    var orig = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{4} });
    defer orig.deinit();

    var stream_buf: [1024]u8 = undefined;
    var stream = @import("stream.zig").MemoryStream.init(&stream_buf);

    try writeToStream(orig, &stream);

    var read_stream = @import("stream.zig").MemoryStream.init(stream.getWritten());
    read_stream.written = stream.written;

    var loaded = try readFromStream(allocator, &read_stream);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 1), loaded.ndim);
    try std.testing.expectEqual(@as(usize, 4), loaded.shape_dims[0]);
    try std.testing.expectEqual(DType.f64, loaded.dtype);

    for (0..4) |i| {
        const v = try loaded.get(f64, &.{i});
        try std.testing.expectEqual(data[i], v);
    }
}

test "NZIG round-trip 2D and 3D arrays" {
    const allocator = std.testing.allocator;

    // 2D Array
    const data2d = [_]i32{ 10, 20, 30, 40, 50, 60 };
    var orig2d = try fromSlice(allocator, i32, .{ .data = &data2d, .shape = &.{ 2, 3 } });
    defer orig2d.deinit();

    var buf2d: [1024]u8 = undefined;
    var stream2d = @import("stream.zig").MemoryStream.init(&buf2d);
    try writeToStream(orig2d, &stream2d);

    var read_stream2d = @import("stream.zig").MemoryStream.init(stream2d.getWritten());
    read_stream2d.written = stream2d.written;

    var loaded2d = try readFromStream(allocator, &read_stream2d);
    defer loaded2d.deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded2d.ndim);
    try std.testing.expectEqual(@as(usize, 2), loaded2d.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 3), loaded2d.shape_dims[1]);
    try std.testing.expectEqual(DType.i32, loaded2d.dtype);
    try std.testing.expectEqual(@as(i32, 50), try loaded2d.get(i32, &.{ 1, 1 }));

    // 3D Array
    const data3d = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var orig3d = try fromSlice(allocator, u8, .{ .data = &data3d, .shape = &.{ 2, 2, 2 } });
    defer orig3d.deinit();

    var buf3d: [1024]u8 = undefined;
    var stream3d = @import("stream.zig").MemoryStream.init(&buf3d);
    try writeToStream(orig3d, &stream3d);

    var read_stream3d = @import("stream.zig").MemoryStream.init(stream3d.getWritten());
    read_stream3d.written = stream3d.written;

    var loaded3d = try readFromStream(allocator, &read_stream3d);
    defer loaded3d.deinit();

    try std.testing.expectEqual(@as(usize, 3), loaded3d.ndim);
    try std.testing.expectEqual(@as(u8, 7), try loaded3d.get(u8, &.{ 1, 1, 0 }));
}

test "NZIG round-trip 0D scalar and empty arrays" {
    const allocator = std.testing.allocator;

    // 0D Scalar
    const scalar_val = [_]f32{42.5};
    var orig0d = try fromSlice(allocator, f32, .{ .data = &scalar_val, .shape = &.{} });
    defer orig0d.deinit();

    var buf0d: [1024]u8 = undefined;
    var stream0d = @import("stream.zig").MemoryStream.init(&buf0d);
    try writeToStream(orig0d, &stream0d);

    var read_stream0d = @import("stream.zig").MemoryStream.init(stream0d.getWritten());
    read_stream0d.written = stream0d.written;

    var loaded0d = try readFromStream(allocator, &read_stream0d);
    defer loaded0d.deinit();

    try std.testing.expectEqual(@as(usize, 0), loaded0d.ndim);
    try std.testing.expectEqual(@as(f32, 42.5), try loaded0d.get(f32, &.{}));

    // Empty array
    var empty_arr = try empty(allocator, .{ .shape = &.{ 3, 0 }, .dtype = .i64 });
    defer empty_arr.deinit();

    var buf_empty: [1024]u8 = undefined;
    var stream_empty = @import("stream.zig").MemoryStream.init(&buf_empty);
    try writeToStream(empty_arr, &stream_empty);

    var read_stream_empty = @import("stream.zig").MemoryStream.init(stream_empty.getWritten());
    read_stream_empty.written = stream_empty.written;

    var loaded_empty = try readFromStream(allocator, &read_stream_empty);
    defer loaded_empty.deinit();

    try std.testing.expectEqual(@as(usize, 0), loaded_empty.elementCount());
    try std.testing.expectEqual(@as(usize, 2), loaded_empty.ndim);
    try std.testing.expectEqual(@as(usize, 3), loaded_empty.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 0), loaded_empty.shape_dims[1]);
}

test "NZIG all supported dtypes round-trip" {
    const allocator = std.testing.allocator;

    // Test bool
    {
        const b_data = [_]bool{ true, false, true };
        var b_arr = try fromSlice(allocator, bool, .{ .data = &b_data, .shape = &.{3} });
        defer b_arr.deinit();

        var s_buf: [512]u8 = undefined;
        var s = @import("stream.zig").MemoryStream.init(&s_buf);
        try writeToStream(b_arr, &s);

        var rs = @import("stream.zig").MemoryStream.init(s.getWritten());
        rs.written = s.written;
        var b_loaded = try readFromStream(allocator, &rs);
        defer b_loaded.deinit();

        try std.testing.expectEqual(true, try b_loaded.get(bool, &.{0}));
        try std.testing.expectEqual(false, try b_loaded.get(bool, &.{1}));
    }

    // Test integers: i16, u32, i64, u64
    {
        const i16_data = [_]i16{ -300, 300 };
        var i16_arr = try fromSlice(allocator, i16, .{ .data = &i16_data, .shape = &.{2} });
        defer i16_arr.deinit();

        var s_buf: [512]u8 = undefined;
        var s = @import("stream.zig").MemoryStream.init(&s_buf);
        try writeToStream(i16_arr, &s);

        var rs = @import("stream.zig").MemoryStream.init(s.getWritten());
        rs.written = s.written;
        var i16_loaded = try readFromStream(allocator, &rs);
        defer i16_loaded.deinit();

        try std.testing.expectEqual(@as(i16, -300), try i16_loaded.get(i16, &.{0}));
        try std.testing.expectEqual(@as(i16, 300), try i16_loaded.get(i16, &.{1}));
    }
}

test "NZIG non-contiguous view and transpose round-trip" {
    const allocator = std.testing.allocator;

    // 2x3 array: [[1, 2, 3], [4, 5, 6]]
    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var orig = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 3 } });
    defer orig.deinit();

    // Transpose -> 3x2 view (non-contiguous)
    const transposed = try @import("../manip/transpose.zig").transpose(orig, .{});
    try std.testing.expect(!transposed.isContiguous());

    var stream_buf: [1024]u8 = undefined;
    var stream = @import("stream.zig").MemoryStream.init(&stream_buf);
    try writeToStream(transposed, &stream);

    var read_stream = @import("stream.zig").MemoryStream.init(stream.getWritten());
    read_stream.written = stream.written;

    var loaded = try readFromStream(allocator, &read_stream);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 3), loaded.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 2), loaded.shape_dims[1]);
    try std.testing.expect(loaded.isContiguous());

    // Verify logical values:
    // transposed[0, 0] = 1, transposed[0, 1] = 4
    // transposed[1, 0] = 2, transposed[1, 1] = 5
    // transposed[2, 0] = 3, transposed[2, 1] = 6
    try std.testing.expectEqual(@as(f64, 1), try loaded.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 4), try loaded.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 2), try loaded.get(f64, &.{ 1, 0 }));
    try std.testing.expectEqual(@as(f64, 5), try loaded.get(f64, &.{ 1, 1 }));
    try std.testing.expectEqual(@as(f64, 3), try loaded.get(f64, &.{ 2, 0 }));
    try std.testing.expectEqual(@as(f64, 6), try loaded.get(f64, &.{ 2, 1 }));
}

test "NZIG deterministic output bytes" {
    const allocator = std.testing.allocator;
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    var arr1 = try fromSlice(allocator, f32, .{ .data = &data, .shape = &.{3} });
    defer arr1.deinit();
    var arr2 = try fromSlice(allocator, f32, .{ .data = &data, .shape = &.{3} });
    defer arr2.deinit();

    var buf1: [512]u8 = undefined;
    var stream1 = @import("stream.zig").MemoryStream.init(&buf1);
    try writeToStream(arr1, &stream1);

    var buf2: [512]u8 = undefined;
    var stream2 = @import("stream.zig").MemoryStream.init(&buf2);
    try writeToStream(arr2, &stream2);

    try std.testing.expectEqualSlices(u8, stream1.getWritten(), stream2.getWritten());
}

test "NZIG corruption tests: bad magic, version, flags, dtype, rank, payload" {
    const allocator = std.testing.allocator;

    var valid_hdr = [_]u8{0} ** HEADER_SIZE;
    @memcpy(valid_hdr[0..8], &MAGIC);
    valid_hdr[8] = VERSION_MAJOR;
    valid_hdr[9] = VERSION_MINOR;
    std.mem.writeInt(u16, valid_hdr[10..12], 0, .little);
    std.mem.writeInt(u16, valid_hdr[12..14], 10, .little); // f32
    valid_hdr[14] = 0; // little
    valid_hdr[15] = 0; // C-order
    valid_hdr[16] = 1; // 1D
    std.mem.writeInt(u64, valid_hdr[24..32], 8, .little); // 8 bytes payload (2 * 4)
    std.mem.writeInt(u64, valid_hdr[32..40], 2, .little); // dim 0 = 2

    const payload = [_]u8{ 0, 0, 128, 63, 0, 0, 0, 64 }; // two f32 values: 1.0, 2.0

    // 1. Bad magic
    {
        var corrupted = valid_hdr;
        corrupted[0] = 0x00;
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidMagic, readFromStream(allocator, &s));
    }

    // 2. Unsupported major version
    {
        var corrupted = valid_hdr;
        corrupted[8] = 99;
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.UnsupportedVersion, readFromStream(allocator, &s));
    }

    // 3. Unsupported flags
    {
        var corrupted = valid_hdr;
        std.mem.writeInt(u16, corrupted[10..12], 1, .little);
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.UnsupportedFlags, readFromStream(allocator, &s));
    }

    // 4. Invalid dtype
    {
        var corrupted = valid_hdr;
        std.mem.writeInt(u16, corrupted[12..14], 9999, .little);
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidDType, readFromStream(allocator, &s));
    }

    // 5. Invalid rank > MAX_RANK
    {
        var corrupted = valid_hdr;
        corrupted[16] = 10;
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidRank, readFromStream(allocator, &s));
    }

    // 6. Payload size mismatch
    {
        var corrupted = valid_hdr;
        std.mem.writeInt(u64, corrupted[24..32], 999, .little);
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidPayloadSize, readFromStream(allocator, &s));
    }

    // 7. Truncated header
    {
        var truncated: [64]u8 = undefined;
        @memcpy(truncated[0..64], valid_hdr[0..64]);
        var s = @import("stream.zig").MemoryStream.init(&truncated);
        s.written = 64;
        try std.testing.expectError(IoError.TruncatedHeader, readFromStream(allocator, &s));
    }

    // 8. Truncated payload
    {
        var full_buf: [HEADER_SIZE + 4]u8 = undefined;
        @memcpy(full_buf[0..HEADER_SIZE], &valid_hdr);
        @memcpy(full_buf[HEADER_SIZE..][0..4], payload[0..4]); // only 4 of 8 bytes
        var s = @import("stream.zig").MemoryStream.init(&full_buf);
        s.written = HEADER_SIZE + 4;
        try std.testing.expectError(IoError.TruncatedPayload, readFromStream(allocator, &s));
    }

    // 9. Invalid byte order
    {
        var corrupted = valid_hdr;
        corrupted[14] = 5;
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidByteOrder, readFromStream(allocator, &s));
    }

    // 10. Invalid memory order
    {
        var corrupted = valid_hdr;
        corrupted[15] = 1;
        var s = @import("stream.zig").MemoryStream.init(&corrupted);
        s.written = HEADER_SIZE;
        try std.testing.expectError(IoError.InvalidOrder, readFromStream(allocator, &s));
    }
}

test "NZIG writeFile and readFile filesystem round-trip" {
    const allocator = std.testing.allocator;
    const data = [_]f64{ 10.5, -20.25, 30.125 };
    var orig = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{3} });
    defer orig.deinit();

    const tmp_path = "test_array_roundtrip.nzig";
    try writeFile(allocator, tmp_path, orig);
    defer {
        var io_threaded: std.Io.Threaded = .init(allocator, .{});
        defer io_threaded.deinit();
        const io = io_threaded.io();
        std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
    }

    var loaded = try readFile(allocator, tmp_path);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 1), loaded.ndim);
    try std.testing.expectEqual(@as(usize, 3), loaded.shape_dims[0]);
    try std.testing.expectEqual(data[0], try loaded.get(f64, &.{0}));
    try std.testing.expectEqual(data[1], try loaded.get(f64, &.{1}));
    try std.testing.expectEqual(data[2], try loaded.get(f64, &.{2}));
}
