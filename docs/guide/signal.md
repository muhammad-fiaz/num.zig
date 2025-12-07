# Signal Processing

`num.zig` provides signal processing capabilities in the `num.signal` module.

## Convolution

Convolution is a fundamental operation in signal processing.

```zig
const num = @import("num");

var a = ...; // [1, 2, 3]
var v = ...; // [0, 1, 0.5]

// Full convolution
var conv = try num.signal.convolve(allocator, f32, a, v, .full);
```

Supported modes:
- `.full`: The output is the full discrete linear convolution.
- `.valid`: The output consists only of those elements that do not rely on zero-padding.
- `.same`: The output is the same size as the first input.

## Correlation

Cross-correlation is also supported via `num.signal.correlate`.

```zig
var corr = try num.signal.correlate(allocator, f32, a, v, .full);
```
