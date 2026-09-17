# Fast Fourier Transform (`num.fft`)

`num.zig` implements an in-house Fast Fourier Transform (FFT) engine capable of 1D and multi-dimensional transforms for both real and complex data.

---

## 1. 1D FFT and IFFT

Transforms complex time-domain signals into frequency spectra using an optimized Radix-2 Cooley-Tukey algorithm (with fallback to direct DFT for arbitrary prime lengths):

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create 8 complex samples
    const signal = [_]Complex(f64){
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 1.0, .im = 0.0 },
        .{ .re = 0.0, .im = 0.0 },
        .{ .re = 0.0, .im = 0.0 },
        .{ .re = 0.0, .im = 0.0 },
        .{ .re = 0.0, .im = 0.0 },
    };

    var input = try num.fromSlice(allocator, Complex(f64), .{
        .data = &signal,
        .shape = &.{8},
    });
    defer input.deinit();

    // Forward FFT
    var spectrum = try num.fft.fft(allocator, f64, &input);
    defer spectrum.deinit();

    // Inverse FFT
    var reconstructed = try num.fft.ifft(allocator, f64, &spectrum);
    defer reconstructed.deinit();
}
```

---

## 2. Real-Valued FFT (`rfft`, `irfft`)

For strictly real-valued input signals, `rfft` exploits Hermitian symmetry to compute only the positive frequencies $N/2 + 1$, cutting computation time and storage in half:

```zig
var real_sig = try num.zeros(allocator, f64, &.{1024});
defer real_sig.deinit();

var half_spectrum = try num.fft.rfft(allocator, f64, &real_sig);
defer half_spectrum.deinit(); // Length: 513
```

---

## 3. Multi-Dimensional FFT (`fftn`, `ifftn`)

Transforms 2D images, 3D volume grids, or N-dimensional tensors across all or specified axes:

```zig
var image = try num.zeros(allocator, Complex(f32), &.{ 256, 256 });
defer image.deinit();

var k_space = try num.fft.fftn(allocator, f32, &image);
defer k_space.deinit();
```

---

## 4. Helper Utilities

- **`fftfreq`**: Computes sample frequencies for given window length and sampling rate.
- **`fftshift`**: Shifts zero-frequency component to the center of the spectrum.
- **`ifftshift`**: Inverts `fftshift`.
