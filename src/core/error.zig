//! Domain-specific error sets for numerical operations.
//!
//! Provides granular, typed error variants across shape resolution, dtype conversions,
//! indexing, linear algebra, and serialization/deserialization.

const std = @import("std");

/// Errors that arise during shape calculation, broadcasting, or dimension inspection.
pub const ShapeError = error{
    /// The specified shapes cannot be broadcast or combined together.
    IncompatibleShapes,
    /// An invalid dimension was specified (e.g. zero or negative size where disallowed).
    InvalidDimension,
    /// An axis parameter is outside the valid range for the array's rank [-rank, rank - 1].
    AxisOutOfBounds,
    /// Shapes do not conform to standard broadcasting rules.
    BroadcastError,
    /// The requested rank exceeds the maximum supported inline rank.
    RankExceeded,
    /// Attempted an operation requiring non-empty array on an array with size 0.
    EmptyArray,
    /// Total element count mismatch when reshaping.
    ReshapeMismatch,
};

/// Errors that arise when converting or operating across incompatible data types.
pub const DTypeError = error{
    /// The requested data types cannot be implicitly or safely promoted.
    IncompatibleDTypes,
    /// An operation does not support the given data type (e.g. floating point ops on integers).
    UnsupportedDType,
    /// Value overflow or precision loss during conversion.
    InvalidConversion,
};

/// Errors that arise during element access or slicing.
pub const IndexError = error{
    /// An index is outside the bounds of the specified dimension.
    IndexOutOfBounds,
    /// A slice specification is invalid.
    InvalidSlice,
    /// A slice step cannot be zero.
    StepCannotBeZero,
    /// Number of indexing keys does not match array rank.
    RankMismatch,
    /// Boolean mask shape does not match target array shape.
    MaskShapeMismatch,
};

/// Errors that arise during linear algebra operations.
pub const LinalgError = error{
    /// Matrix must be square (rows == cols) for this operation.
    MatrixNotSquare,
    /// Matrix is singular and cannot be inverted or solved.
    SingularMatrix,
    /// Matrix is not positive-definite (required for Cholesky decomposition).
    NotPositiveDefinite,
    /// Inner matrix dimensions do not match for multiplication (K of MxK and KxN).
    IncompatibleDimensions,
    /// Algorithm failed to converge.
    ConvergenceFailed,
};

/// Errors that arise during binary or text input/output operations.
pub const IoError = error{
    /// File does not begin with the expected magic bytes.
    InvalidMagic,
    /// File format version is not supported by this version of the library.
    UnsupportedVersion,
    /// The dtype identifier in the file header is invalid or unsupported.
    InvalidDType,
    /// The rank in the file header is invalid or exceeds maximum rank.
    InvalidRank,
    /// The shape dimensions in the file header are invalid or overflow.
    InvalidShape,
    /// The byte order indicator is invalid.
    InvalidByteOrder,
    /// The memory order indicator is invalid.
    InvalidOrder,
    /// The declared payload size does not match calculated element bytes.
    InvalidPayloadSize,
    /// The file or stream ended prematurely before reading the full header.
    TruncatedHeader,
    /// The payload ended prematurely before all expected elements were read.
    TruncatedPayload,
    /// The file header is corrupted or contains invalid configuration flags.
    CorruptHeader,
    /// Format flags contain unsupported features.
    UnsupportedFlags,
    /// Unexpected trailing data after the declared payload.
    UnexpectedTrailingData,
    /// Target array shape does not match shape declared in data stream.
    ShapeMismatch,
    /// Target array dtype does not match dtype declared in data stream.
    DTypeMismatch,
    /// Endianness byte swapping failed or is unsupported.
    EndiannessMismatch,
    /// Delimited text parser encountered an invalid number format.
    ParseError,
    /// Stream ended before expected delimiter or line break.
    EndOfStream,
    /// Underlying write failed to flush.
    WriteFailed,
    /// Requested rank exceeds maximum supported rank for format.
    UnsupportedRank,
    /// General truncated data.
    TruncatedData,
};

/// Composed top-level error set covering all numerical operations.
pub const Error = ShapeError || DTypeError || IndexError || LinalgError || IoError || std.mem.Allocator.Error;

test "error sets composition" {
    const err1: Error = ShapeError.IncompatibleShapes;
    const err2: Error = LinalgError.SingularMatrix;
    const err3: Error = IoError.InvalidMagic;
    const err4: Error = error.OutOfMemory;

    try std.testing.expect(err1 == ShapeError.IncompatibleShapes);
    try std.testing.expect(err2 == LinalgError.SingularMatrix);
    try std.testing.expect(err3 == IoError.InvalidMagic);
    try std.testing.expect(err4 == error.OutOfMemory);
}
