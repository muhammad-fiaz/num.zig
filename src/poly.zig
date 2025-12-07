const std = @import("std");
const core = @import("core.zig");
const linalg = @import("linalg.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Evaluate a polynomial at specific values.
///
/// Arguments:
///     allocator: Allocator.
///     T: Element type.
///     p: Polynomial coefficients (highest degree first).
///     x: Values to evaluate at.
///
/// Returns:
///     New array with evaluated values.
pub fn polyval(allocator: Allocator, comptime T: type, p: NDArray(T), x: NDArray(T)) !NDArray(T) {
    if (p.rank() != 1) return core.Error.RankMismatch;

    var result = try NDArray(T).init(allocator, x.shape);
    errdefer result.deinit();

    // Horner's method
    // p[0]*x^N + p[1]*x^(N-1) + ... + p[N]

    // Iterate over x
    var iter = try core.NdIterator.init(allocator, x.shape);
    defer iter.deinit();

    while (iter.next()) |coords| {
        const val_x = try x.get(coords);
        var res: T = 0;

        for (p.data) |c| {
            res = res * val_x + c;
        }

        try result.set(coords, res);
    }

    return result;
}

/// Add two polynomials.
pub fn polyadd(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T) {
    if (p1.rank() != 1 or p2.rank() != 1) return core.Error.RankMismatch;

    const n1 = p1.size();
    const n2 = p2.size();
    const n = @max(n1, n2);

    var result = try NDArray(T).zeros(allocator, &.{n});

    // Add coefficients, aligning to the right (lowest power last)
    const offset1 = n - n1;
    const offset2 = n - n2;

    for (0..n) |i| {
        var val: T = 0;
        if (i >= offset1) val += p1.data[i - offset1];
        if (i >= offset2) val += p2.data[i - offset2];
        result.data[i] = val;
    }

    return result;
}

/// Subtract two polynomials.
pub fn polysub(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T) {
    if (p1.rank() != 1 or p2.rank() != 1) return core.Error.RankMismatch;

    const n1 = p1.size();
    const n2 = p2.size();
    const n = @max(n1, n2);

    var result = try NDArray(T).zeros(allocator, &.{n});

    const offset1 = n - n1;
    const offset2 = n - n2;

    for (0..n) |i| {
        var val: T = 0;
        if (i >= offset1) val += p1.data[i - offset1];
        if (i >= offset2) val -= p2.data[i - offset2];
        result.data[i] = val;
    }

    return result;
}

/// Multiply two polynomials.
pub fn polymul(allocator: Allocator, comptime T: type, p1: NDArray(T), p2: NDArray(T)) !NDArray(T) {
    // Convolution of coefficients
    const signal = @import("signal.zig");
    return signal.convolve(allocator, T, p1, p2, .full);
}

/// Return the roots of a polynomial with coefficients given in p.
/// The values in the rank-1 array p are coefficients of a polynomial.
/// If the length of p is n+1 then the polynomial is described by:
/// p[0] * x^n + p[1] * x^(n-1) + ... + p[n-1]*x + p[n]
pub fn roots(allocator: Allocator, comptime T: type, p: NDArray(T)) !NDArray(T) {
    if (p.rank() != 1) return core.Error.RankMismatch;
    const n = p.size();
    if (n < 2) return NDArray(T).init(allocator, &.{0});

    // Construct companion matrix
    // The companion matrix of p(x) = c0 x^n + c1 x^(n-1) + ... + cn
    // is an n x n matrix.
    // We first normalize by c0.

    const c0 = p.data[0];
    if (c0 == 0) return core.Error.SingularMatrix; // Leading coefficient zero

    const dim = n - 1;
    var companion = try NDArray(T).zeros(allocator, &.{ dim, dim });
    defer companion.deinit();

    // Fill sub-diagonal with 1s
    for (1..dim) |i| {
        try companion.set(&.{ i, i - 1 }, 1);
    }

    // Fill first row with -ci/c0
    for (0..dim) |i| {
        const val = -p.data[i + 1] / c0;
        try companion.set(&.{ 0, i }, val);
    }

    // Compute eigenvalues
    // Note: eigvals returns real parts if T is real.
    // Ideally we should return complex roots, but for now we return T (real roots approximation or if T is complex).
    return linalg.eigvals(T, allocator, &companion, 100);
}

/// Return the derivative of the specified order of a polynomial.
pub fn polyder(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize) !NDArray(T) {
    if (p.rank() != 1) return core.Error.RankMismatch;
    const n = p.size();

    if (m >= n) {
        return NDArray(T).zeros(allocator, &.{1});
    }

    var result = try p.copy(allocator);

    for (0..m) |_| {
        const current_n = result.size();
        if (current_n <= 1) {
            result.deinit();
            result = try NDArray(T).zeros(allocator, &.{1});
            break;
        }

        var new_coeffs = try NDArray(T).init(allocator, &.{current_n - 1});
        for (0..current_n - 1) |i| {
            const power = @as(T, @floatFromInt(current_n - 1 - i));
            new_coeffs.data[i] = result.data[i] * power;
        }
        result.deinit();
        result = new_coeffs;
    }

    return result;
}

/// Return the antiderivative (indefinite integral) of a polynomial.
pub fn polyint(allocator: Allocator, comptime T: type, p: NDArray(T), m: usize, k: T) !NDArray(T) {
    if (p.rank() != 1) return core.Error.RankMismatch;

    var result = try p.copy(allocator);

    for (0..m) |_| {
        const current_n = result.size();
        var new_coeffs = try NDArray(T).init(allocator, &.{current_n + 1});

        for (0..current_n) |i| {
            const power = @as(T, @floatFromInt(current_n - i));
            new_coeffs.data[i] = result.data[i] / power;
        }
        new_coeffs.data[current_n] = k; // Constant of integration (applied at each step? usually just once at end or 0s. NumPy does 0s usually unless k specified)
        // Here we apply k at the last position. If m > 1, subsequent integrals will integrate this k.

        result.deinit();
        result = new_coeffs;
    }

    return result;
}
