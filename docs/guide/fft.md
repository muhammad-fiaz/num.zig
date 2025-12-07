# Fast Fourier Transform (FFT)

The `num.fft` module provides Fast Fourier Transform operations.

## FFT

Compute the 1D FFT. Input length must be a power of 2.

```zig
const Complex = std.math.Complex;
var res = try num.fft.FFT.fft(allocator, &input_arr);
// Returns NDArray(Complex(f32))
```

## Inverse FFT

Compute the 1D Inverse FFT.

```zig
var original = try num.fft.FFT.ifft(allocator, &res);
```
