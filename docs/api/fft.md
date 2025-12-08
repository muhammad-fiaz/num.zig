# FFT API Reference

The `fft` module provides Fast Fourier Transform functions.

## 1D FFT

### fft

Compute the one-dimensional discrete Fourier Transform.

```zig
pub fn fft(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(Complex(T))
```

### ifft

Compute the one-dimensional inverse discrete Fourier Transform.

```zig
pub fn ifft(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T))
```

## 2D FFT

### fft2

Compute the 2-dimensional discrete Fourier Transform.

```zig
pub fn fft2(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(Complex(T))
```

### ifft2

Compute the 2-dimensional inverse discrete Fourier Transform.

```zig
pub fn ifft2(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T))
```

## N-D FFT

### fftn

Compute the N-dimensional discrete Fourier Transform.

```zig
pub fn fftn(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(Complex(T))
```

### ifftn

Compute the N-dimensional inverse discrete Fourier Transform.

```zig
pub fn ifftn(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T))
```