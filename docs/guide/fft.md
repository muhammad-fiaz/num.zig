# Fast Fourier Transform (`num.fft`)

`num.zig` implements a Fast Fourier Transform engine with 1D transforms along a selected axis, 2D transforms over the last two axes, frequency helpers, and shift utilities. Complex element types reuse Zig 0.16.0 `std.math.Complex` (`num.fft.Complex64` / `num.fft.Complex128`).

---

## 1. 1D FFT and IFFT

Transforms signals along a selected axis (default last axis), reusing the shared Radix-2/DFT kernel with `backward`, `ortho`, and `forward` normalizations:

```zig
var input = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
    .shape = &.{8},
});
defer input.deinit();

// Forward FFT
var spectrum = try num.fft.fft(input, .{});
defer spectrum.deinit();

// Inverse FFT
var reconstructed = try num.fft.ifft(spectrum, .{});
defer reconstructed.deinit();
```

---

## 2. 2D FFT (`fft2`, `ifft2`)

Transforms rank-2+ arrays over the last two axes by reusing the 1D kernel sequentially:

```zig
var image = try num.zeros(allocator, .{ .shape = &.{ 4, 4 }, .dtype = .f64 });
defer image.deinit();

var k_space = try num.fft.fft2(image, .{});
defer k_space.deinit();

var restored = try num.fft.ifft2(k_space, .{});
defer restored.deinit();
```

---

## 3. Helper Utilities

- **`fftfreq`**: DFT sample frequencies for a window length and sampling interval.
- **`rfftfreq`**: Frequencies for real-input transforms (length `n/2 + 1`).
- **`fftshift`**: Shifts zero-frequency component to the center of the spectrum.
- **`ifftshift`**: Inverts `fftshift`.
