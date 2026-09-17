//! num.zig — Native numerical computing library for Zig.

const std = @import("std");

// Core types & errors
pub const Array = @import("core/array.zig").Array;
pub const DType = @import("core/dtype.zig").DType;
pub const Order = @import("core/shape.zig").Order;
pub const Shape = @import("core/shape.zig").Shape;
pub const Strides = @import("core/shape.zig").Strides;
pub const Slice = @import("core/shape.zig").Slice;
pub const MAX_RANK = @import("core/shape.zig").MAX_RANK;
pub const Buffer = @import("core/buffer.zig").Buffer;

pub const Error = @import("core/error.zig").Error;
pub const ShapeError = @import("core/error.zig").ShapeError;
pub const DTypeError = @import("core/error.zig").DTypeError;
pub const IndexError = @import("core/error.zig").IndexError;
pub const LinalgError = @import("core/error.zig").LinalgError;
pub const IoError = @import("core/error.zig").IoError;

// Iterators
pub const iterator = @import("core/iterator.zig");

// Domain Modules
pub const array = struct {
    pub const zeros = @import("core/array.zig").zeros;
    pub const ones = @import("core/array.zig").ones;
    pub const full = @import("core/array.zig").full;
    pub const empty = @import("core/array.zig").empty;
    pub const scalar = @import("core/array.zig").scalar;
    pub const arange = @import("core/array.zig").arange;
    pub const linspace = @import("core/array.zig").linspace;
    pub const logspace = @import("core/array.zig").logspace;
    pub const geomspace = @import("core/array.zig").geomspace;
    pub const eye = @import("core/array.zig").eye;
    pub const identity = @import("core/array.zig").identity;
    pub const diag = @import("core/array.zig").diag;
    pub const triu = @import("core/array.zig").triu;
    pub const tril = @import("core/array.zig").tril;
    pub const fromSlice = @import("core/array.zig").fromSlice;
};

// Top-level creation and linalg shortcuts
pub const zeros = array.zeros;
pub const ones = array.ones;
pub const full = array.full;
pub const empty = array.empty;
pub const scalar = array.scalar;
pub const arange = array.arange;
pub const linspace = array.linspace;
pub const logspace = array.logspace;
pub const geomspace = array.geomspace;
pub const eye = array.eye;
pub const identity = array.identity;
pub const diag = array.diag;
pub const triu = array.triu;
pub const tril = array.tril;
pub const fromSlice = array.fromSlice;
pub const dot = linalg.dot;
pub const matmul = linalg.matmul;
pub const equal = ops.equal;
pub const notEqual = ops.notEqual;
pub const all = reduce.all;
pub const any = reduce.any;

pub const manip = struct {
    pub const reshape = @import("manip/reshape.zig").reshape;
    pub const ravel = @import("manip/reshape.zig").ravel;
    pub const flatten = @import("manip/reshape.zig").flatten;
    pub const squeeze = @import("manip/reshape.zig").squeeze;
    pub const expandDims = @import("manip/reshape.zig").expandDims;
    pub const atleast1d = @import("manip/reshape.zig").atleast1d;
    pub const atleast2d = @import("manip/reshape.zig").atleast2d;
    pub const atleast3d = @import("manip/reshape.zig").atleast3d;
    pub const unravelIndex = @import("manip/reshape.zig").unravelIndex;
    pub const ravelIndex = @import("manip/reshape.zig").ravelIndex;
    pub const indices = @import("manip/reshape.zig").indices;
    pub const transpose = @import("manip/transpose.zig").transpose;
    pub const swapAxes = @import("manip/transpose.zig").swapAxes;
    pub const moveAxis = @import("manip/transpose.zig").moveAxis;
    pub const flip = @import("manip/transpose.zig").flip;
    pub const roll = @import("manip/transpose.zig").roll;
    pub const slice = @import("manip/slice.zig").slice;
    pub const concat = @import("manip/concat.zig").concat;
    pub const stack = @import("manip/concat.zig").stack;
    pub const hstack = @import("manip/concat.zig").hstack;
    pub const vstack = @import("manip/concat.zig").vstack;
    pub const split = @import("manip/concat.zig").split;
    pub const append = @import("manip/concat.zig").append;
    pub const insert = @import("manip/concat.zig").insert;
    pub const delete = @import("manip/concat.zig").delete;
    pub const tile = @import("manip/concat.zig").tile;
    pub const repeat = @import("manip/concat.zig").repeat;
    pub const pad = @import("manip/pad.zig").pad;
    pub const PadMode = @import("manip/pad.zig").PadMode;
};

pub const broadcast = struct {
    pub const broadcastTo = @import("ops/broadcast.zig").broadcastTo;
    pub const broadcast2 = @import("ops/broadcast.zig").broadcast2;
    pub const broadcastShapes = @import("core/shape.zig").broadcastShapes;
};

pub const ops = struct {
    pub const add = @import("ops/elementwise.zig").add;
    pub const subtract = @import("ops/elementwise.zig").subtract;
    pub const multiply = @import("ops/elementwise.zig").multiply;
    pub const divide = @import("ops/elementwise.zig").divide;
    pub const pow = @import("ops/elementwise.zig").pow;
    pub const negate = @import("ops/elementwise.zig").negate;
    pub const positive = @import("ops/elementwise.zig").positive;
    pub const abs = @import("ops/elementwise.zig").abs;
    pub const sqrt = @import("ops/elementwise.zig").sqrt;
    pub const exp = @import("ops/elementwise.zig").exp;
    pub const log = @import("ops/elementwise.zig").log;
    pub const log2 = @import("ops/elementwise.zig").log2;
    pub const log10 = @import("ops/elementwise.zig").log10;
    pub const sin = @import("ops/elementwise.zig").sin;
    pub const cos = @import("ops/elementwise.zig").cos;
    pub const tan = @import("ops/elementwise.zig").tan;
    pub const sinh = @import("ops/elementwise.zig").sinh;
    pub const cosh = @import("ops/elementwise.zig").cosh;
    pub const tanh = @import("ops/elementwise.zig").tanh;
    pub const floor = @import("ops/elementwise.zig").floor;
    pub const ceil = @import("ops/elementwise.zig").ceil;
    pub const round = @import("ops/elementwise.zig").round;
    pub const trunc = @import("ops/elementwise.zig").trunc;
    pub const sign = @import("ops/elementwise.zig").sign;
    pub const cbrt = @import("ops/elementwise.zig").cbrt;
    pub const square = @import("ops/elementwise.zig").square;
    pub const expm1 = @import("ops/elementwise.zig").expm1;
    pub const log1p = @import("ops/elementwise.zig").log1p;
    pub const asin = @import("ops/elementwise.zig").asin;
    pub const acos = @import("ops/elementwise.zig").acos;
    pub const atan = @import("ops/elementwise.zig").atan;
    pub const atan2 = @import("ops/elementwise.zig").atan2;
    pub const hypot = @import("ops/elementwise.zig").hypot;
    pub const asinh = @import("ops/elementwise.zig").asinh;
    pub const acosh = @import("ops/elementwise.zig").acosh;
    pub const atanh = @import("ops/elementwise.zig").atanh;
    pub const gamma = @import("ops/elementwise.zig").gamma;
    pub const lgamma = @import("ops/elementwise.zig").lgamma;
    pub const erf = @import("ops/elementwise.zig").erf;
    pub const erfc = @import("ops/elementwise.zig").erfc;
    pub const degreesToRadians = @import("ops/elementwise.zig").degreesToRadians;
    pub const radiansToDegrees = @import("ops/elementwise.zig").radiansToDegrees;
    pub const reciprocal = @import("ops/elementwise.zig").reciprocal;
    pub const exp2 = @import("ops/elementwise.zig").exp2;
    pub const remainder = @import("ops/elementwise.zig").remainder;
    pub const minimum = @import("ops/elementwise.zig").minimum;
    pub const maximum = @import("ops/elementwise.zig").maximum;
    pub const clip = @import("ops/elementwise.zig").clip;
    pub const where = @import("ops/elementwise.zig").where;

    // Deliberate shorthand aliases mapping directly to canonical functions
    pub const sub = subtract;
    pub const mul = multiply;
    pub const div = divide;
    pub const rem = remainder;
    pub const mod = remainder;
    pub const power = pow;
    pub const absolute = abs;
    pub const negative = negate;

    // Comparisons & Predicates
    pub const equal = @import("ops/compare.zig").equal;
    pub const notEqual = @import("ops/compare.zig").notEqual;
    pub const less = @import("ops/compare.zig").less;
    pub const lessEqual = @import("ops/compare.zig").lessEqual;
    pub const greater = @import("ops/compare.zig").greater;
    pub const greaterEqual = @import("ops/compare.zig").greaterEqual;
    pub const logicalAnd = @import("ops/compare.zig").logicalAnd;
    pub const logicalOr = @import("ops/compare.zig").logicalOr;
    pub const logicalNot = @import("ops/compare.zig").logicalNot;
    pub const logicalXor = @import("ops/compare.zig").logicalXor;
    pub const isNaN = @import("ops/compare.zig").isNaN;
    pub const isInf = @import("ops/compare.zig").isInf;
    pub const isFinite = @import("ops/compare.zig").isFinite;
    pub const isClose = @import("ops/compare.zig").isClose;
    pub const allClose = @import("ops/compare.zig").allClose;

    // Integer bitwise operations (integer dtypes only, broadcast)
    pub const bitwiseAnd = @import("ops/bitwise.zig").bitwiseAnd;
    pub const bitwiseOr = @import("ops/bitwise.zig").bitwiseOr;
    pub const bitwiseXor = @import("ops/bitwise.zig").bitwiseXor;
    pub const bitwiseNot = @import("ops/bitwise.zig").bitwiseNot;
    pub const leftShift = @import("ops/bitwise.zig").leftShift;
    pub const rightShift = @import("ops/bitwise.zig").rightShift;
    pub const bitCount = @import("ops/bitwise.zig").bitCount;
    pub const clz = @import("ops/bitwise.zig").clz;
    pub const ctz = @import("ops/bitwise.zig").ctz;
    pub const leadingZeros = @import("ops/bitwise.zig").leadingZeros;
    pub const trailingZeros = @import("ops/bitwise.zig").trailingZeros;
    pub const popcount = @import("ops/bitwise.zig").popcount;

    // Complex helpers
    pub const conj = @import("ops/complex.zig").conj;
    pub const conjugate = @import("ops/complex.zig").conjugate;
    pub const conjTranspose = @import("ops/complex.zig").conjTranspose;
    pub const real = @import("ops/complex.zig").real;
    pub const imag = @import("ops/complex.zig").imag;
    pub const magnitude = @import("ops/complex.zig").magnitude;
    pub const phase = @import("ops/complex.zig").phase;
};

pub const reduce = struct {
    pub const sum = @import("ops/reduce.zig").sum;
    pub const prod = @import("ops/reduce.zig").prod;
    pub const mean = @import("ops/reduce.zig").mean;
    pub const median = @import("ops/reduce.zig").median;
    pub const variance = @import("ops/reduce.zig").variance;
    pub const stdDev = @import("ops/reduce.zig").stdDev;
    pub const min = @import("ops/reduce.zig").min;
    pub const max = @import("ops/reduce.zig").max;
    pub const argmin = @import("ops/reduce.zig").argmin;
    pub const argmax = @import("ops/reduce.zig").argmax;
    pub const all = @import("ops/reduce.zig").all;
    pub const any = @import("ops/reduce.zig").any;
    pub const cumsum = @import("ops/reduce.zig").cumsum;
    pub const cumprod = @import("ops/reduce.zig").cumprod;
    pub const cummin = @import("ops/reduce.zig").cummin;
    pub const cummax = @import("ops/reduce.zig").cummax;
    pub const countNonzero = @import("ops/reduce.zig").countNonzero;
    pub const diff = @import("ops/reduce.zig").diff;
};

pub const linalg = struct {
    pub const matmul = @import("linalg/matmul.zig").matmul;
    pub const dot = @import("linalg/matmul.zig").dot;
    pub const inner = @import("linalg/matmul.zig").inner;
    pub const outer = @import("linalg/matmul.zig").outer;
    pub const kron = @import("linalg/matmul.zig").kron;
    pub const norm = @import("linalg/norm.zig").norm;
    pub const lu = @import("linalg/decompose.zig").lu;
    pub const qr = @import("linalg/decompose.zig").qr;
    pub const cholesky = @import("linalg/decompose.zig").cholesky;
    pub const solve = @import("linalg/solve.zig").solve;
    pub const solveTriangular = @import("linalg/solve.zig").solveTriangular;
    pub const solveSpd = @import("linalg/solve.zig").solveSpd;
    pub const lstsq = @import("linalg/solve.zig").lstsq;
    pub const inv = @import("linalg/solve.zig").inv;
    pub const det = @import("linalg/solve.zig").det;
    pub const slogdet = @import("linalg/solve.zig").slogdet;
    pub const SlogdetResult = @import("linalg/solve.zig").SlogdetResult;
    pub const trace = @import("linalg/solve.zig").trace;
    pub const matrixRank = @import("linalg/solve.zig").matrixRank;
    pub const matrixPower = @import("linalg/solve.zig").matrixPower;
    pub const pinv = @import("linalg/solve.zig").pinv;
    pub const eig = @import("linalg/eigen.zig").eig;
    pub const eigvals = @import("linalg/eigen.zig").eigvals;
    pub const svd = @import("linalg/svd.zig").svd;
};

pub const fft = struct {
    pub const fft = @import("linalg/fft.zig").fft;
    pub const ifft = @import("linalg/fft.zig").ifft;
    pub const fft2 = @import("linalg/fft.zig").fft2;
    pub const ifft2 = @import("linalg/fft.zig").ifft2;
    pub const Norm = @import("linalg/fft.zig").Norm;
    pub const fftfreq = @import("linalg/fft.zig").fftfreq;
    pub const rfftfreq = @import("linalg/fft.zig").rfftfreq;
    pub const fftshift = @import("linalg/fft.zig").fftshift;
    pub const ifftshift = @import("linalg/fft.zig").ifftshift;
    pub const Complex64 = @import("linalg/fft.zig").Complex64;
    pub const Complex128 = @import("linalg/fft.zig").Complex128;
};

pub const sparse = struct {
    pub const CsrMatrix = @import("core/sparse.zig").CsrMatrix;
    pub const CscMatrix = @import("core/sparse.zig").CscMatrix;
    pub const cg = @import("linalg/sparse_solve.zig").cg;
    pub const gmres = @import("linalg/sparse_solve.zig").gmres;
    pub const SparseSolveResult = @import("linalg/sparse_solve.zig").SparseSolveResult;
};

pub const parallel = struct {
    pub const run = @import("ops/parallel.zig").run;
};

pub const random = struct {
    pub const Prng = @import("random/engine.zig").Prng;
    pub const uniform = @import("random/distributions.zig").uniform;
    pub const rand = @import("random/distributions.zig").rand;
    pub const normal = @import("random/distributions.zig").normal;
    pub const randn = @import("random/distributions.zig").randn;
    pub const integers = @import("random/distributions.zig").integers;
    pub const choice = @import("random/distributions.zig").choice;
    pub const shuffle = @import("random/distributions.zig").shuffle;
    pub const permutation = @import("random/distributions.zig").permutation;
};

pub const stats = struct {
    pub const mean = @import("stats/describe.zig").mean;
    pub const min = @import("stats/describe.zig").min;
    pub const max = @import("stats/describe.zig").max;
    pub const range = @import("stats/describe.zig").range;
    pub const variance = @import("stats/describe.zig").variance;
    pub const stdDev = @import("stats/describe.zig").stdDev;
    pub const median = @import("stats/describe.zig").median;
    pub const quantile = @import("stats/describe.zig").quantile;
    pub const quantileWithOptions = @import("stats/describe.zig").quantileWithOptions;
    pub const QuantileMethod = @import("stats/describe.zig").QuantileMethod;
    pub const percentile = @import("stats/describe.zig").percentile;
    pub const covariance = @import("stats/correlate.zig").covariance;
    pub const corrcoef = @import("stats/correlate.zig").corrcoef;
    pub const histogram = @import("stats/correlate.zig").histogram;
};

pub const sort = struct {
    pub const sort = @import("sort/ordering.zig").sort;
    pub const sorted = @import("sort/ordering.zig").sorted;
    pub const argsort = @import("sort/ordering.zig").argsort;
    pub const searchSorted = @import("sort/search.zig").searchSorted;
    pub const unique = @import("sort/search.zig").unique;
    pub const flatNonzero = @import("sort/search.zig").flatNonzero;
    pub const nonzero = @import("sort/search.zig").nonzero;
    pub const argwhere = @import("sort/search.zig").argwhere;
    pub const intersect1d = @import("sort/search.zig").intersect1d;
    pub const union1d = @import("sort/search.zig").union1d;
    pub const setdiff1d = @import("sort/search.zig").setdiff1d;
    pub const isin = @import("sort/search.zig").isin;
};

pub const io = struct {
    pub const nzig = @import("io/nzig.zig");
    pub const text = @import("io/text.zig");
    pub const stream = @import("io/stream.zig");

    // Primary native NZIG serialization
    pub const writeFile = @import("io/nzig.zig").writeFile;
    pub const readFile = @import("io/nzig.zig").readFile;
    pub const writeToStream = @import("io/nzig.zig").writeToStream;
    pub const readFromStream = @import("io/nzig.zig").readFromStream;

    // Delimited text I/O
    pub const savetxt = @import("io/text.zig").savetxt;
    pub const loadtxt = @import("io/text.zig").loadtxt;
    pub const savetxtWriter = @import("io/text.zig").savetxtWriter;
    pub const loadtxtReader = @import("io/text.zig").loadtxtReader;

    // Formatting and stream utilities
    pub const formatArray = @import("strops/format.zig").formatArray;
    pub const MemoryStream = @import("io/stream.zig").MemoryStream;
};

// Top-level shortcuts for native NZIG serialization
pub const save = io.writeFile;
pub const load = io.readFile;

pub const poly = struct {
    pub const val = @import("poly/eval.zig").val;
    pub const fit = @import("poly/fit.zig").fit;
    pub const der = @import("poly/eval.zig").der;
    pub const integ = @import("poly/eval.zig").integ;
    pub const roots = @import("poly/roots.zig").roots;
    pub const add = @import("poly/roots.zig").add;
    pub const sub = @import("poly/roots.zig").sub;
    pub const mul = @import("poly/roots.zig").mul;
};

test {
    std.testing.refAllDecls(@This());
    _ = @import("core/error.zig");
    _ = @import("core/dtype.zig");
    _ = @import("core/shape.zig");
    _ = @import("core/buffer.zig");
    _ = @import("core/iterator.zig");
    _ = @import("core/array.zig");
    _ = @import("ops/broadcast.zig");
    _ = @import("ops/elementwise.zig");
    _ = @import("ops/bitwise.zig");
    _ = @import("ops/complex.zig");
    _ = @import("ops/compare.zig");
    _ = @import("ops/reduce.zig");
    _ = @import("manip/reshape.zig");
    _ = @import("manip/transpose.zig");
    _ = @import("manip/slice.zig");
    _ = @import("manip/concat.zig");
    _ = @import("manip/pad.zig");
    _ = @import("linalg/matmul.zig");
    _ = @import("linalg/norm.zig");
    _ = @import("linalg/decompose.zig");
    _ = @import("linalg/solve.zig");
    _ = @import("linalg/eigen.zig");
    _ = @import("linalg/svd.zig");
    _ = @import("linalg/fft.zig");
    _ = @import("core/sparse.zig");
    _ = @import("random/engine.zig");
    _ = @import("random/distributions.zig");
    _ = @import("stats/describe.zig");
    _ = @import("stats/correlate.zig");
    _ = @import("sort/ordering.zig");
    _ = @import("sort/search.zig");
    _ = @import("io/nzig.zig");
    _ = @import("io/text.zig");
    _ = @import("io/stream.zig");
    _ = @import("strops/format.zig");
    _ = @import("linalg/sparse_solve.zig");
    _ = @import("ops/parallel.zig");
    _ = @import("poly/eval.zig");
    _ = @import("poly/fit.zig");
    _ = @import("poly/roots.zig");
}

test "end-to-end library integration" {
    const allocator = std.testing.allocator;

    // 1. Array creation
    var a = try arange(allocator, .{ .start = 0, .stop = 6, .step = 1, .dtype = .f64 });
    defer a.deinit();

    var a_2d = try manip.reshape(a, .{ .shape = &.{ 2, 3 } });
    defer a_2d.deinit();

    // 2. Elementwise operations and reductions
    var ones_arr = try ones(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
    defer ones_arr.deinit();

    var b = try ops.add(a_2d, ones_arr, .{});
    defer b.deinit();

    var s = try reduce.sum(b, .{});
    defer s.deinit();
    // Sum of {1, 2, 3, 4, 5, 6} is 21.0
    try std.testing.expectApproxEqAbs(@as(f64, 21.0), try s.get(f64, &.{}), 1e-5);

    // 3. Matrix multiplication
    var b_t = try manip.transpose(b, .{});
    defer b_t.deinit();

    var c = try matmul(b, b_t, .{});
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 2), c.shape_dims[0]);
    try std.testing.expectEqual(@as(usize, 2), c.shape_dims[1]);

    // 4. Statistics and sorting
    var med = try stats.median(a, .{});
    defer med.deinit();
    // Median of 0, 1, 2, 3, 4, 5 is 2.5
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), try med.get(f64, &.{}), 1e-5);

    // 5. Polynomial fitting and evaluation
    const x_vals = [_]f64{ 1.0, 2.0, 3.0 };
    const y_vals = [_]f64{ 3.0, 5.0, 7.0 };
    var x = try fromSlice(allocator, f64, .{ .data = &x_vals, .shape = &.{3} });
    defer x.deinit();
    var y = try fromSlice(allocator, f64, .{ .data = &y_vals, .shape = &.{3} });
    defer y.deinit();

    var fit_res = try poly.fit(x, y, 1);
    defer fit_res.deinit();
    // y = 2x + 1
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), try fit_res.get(f64, &.{0}), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try fit_res.get(f64, &.{1}), 1e-4);

    // 6. Native NZIG binary serialization stream
    var buf: [1024]u8 = undefined;
    var ms = io.MemoryStream.init(&buf);
    try io.writeToStream(c, &ms);

    var rs = io.MemoryStream.init(ms.getWritten());
    rs.written = ms.written;
    var loaded = try io.readFromStream(allocator, &rs);
    defer loaded.deinit();

    try std.testing.expectEqual(c.ndim, loaded.ndim);
    try std.testing.expectApproxEqAbs(try c.get(f64, &.{ 0, 0 }), try loaded.get(f64, &.{ 0, 0 }), 1e-5);
}
