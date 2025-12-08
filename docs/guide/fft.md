# Fast Fourier Transform

The `fft` module provides efficient Fast Fourier Transform implementations for 1D, 2D, and N-dimensional arrays. FFT is essential for signal processing, frequency analysis, and solving partial differential equations.

## 1D FFT (Forward)

Compute the 1D FFT of a real signal:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple signal: [1, 2, 3, 4]
    const shape = [_]usize{4};
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var signal = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // Compute FFT
    var spectrum = try num.fft.fft(allocator, &signal);
    defer spectrum.deinit(allocator);
    
    std.debug.print("FFT result:\n", .{});
    for (0..4) |i| {
        std.debug.print("  [{d:.2} + {d:.2}i]\n", .{
            spectrum.data[i].real,
            spectrum.data[i].imag,
        });
    }
}
// Output:
// FFT result:
//   [10.00 + 0.00i]
//   [-2.00 + 2.00i]
//   [-2.00 + 0.00i]
//   [-2.00 - 2.00i]
```

## 1D IFFT (Inverse)

Recover the original signal from its FFT:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{4};
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var signal = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // Forward FFT
    var spectrum = try num.fft.fft(allocator, &signal);
    defer spectrum.deinit(allocator);
    
    // Inverse FFT
    var reconstructed = try num.fft.ifft(allocator, &spectrum);
    defer reconstructed.deinit(allocator);
    
    std.debug.print("Reconstructed signal:\n", .{});
    for (0..4) |i| {
        std.debug.print("  {d:.2}\n", .{reconstructed.data[i].real});
    }
}
// Output:
// Reconstructed signal:
//   1.00
//   2.00
//   3.00
//   4.00
```

## 2D FFT

Compute the 2D FFT for image or matrix data:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2x2 complex matrix
    const shape = [_]usize{ 2, 2 };
    var data = [_]Complex(f32){
        Complex(f32){ .real = 1.0, .imag = 0.0 },
        Complex(f32){ .real = 2.0, .imag = 0.0 },
        Complex(f32){ .real = 3.0, .imag = 0.0 },
        Complex(f32){ .real = 4.0, .imag = 0.0 },
    };
    var matrix = num.core.NDArray(Complex(f32)).init(&shape, &data);
    
    // Compute 2D FFT
    var spectrum2d = try num.fft.fft2(allocator, &matrix);
    defer spectrum2d.deinit(allocator);
    
    std.debug.print("2D FFT result:\n", .{});
    for (0..2) |i| {
        for (0..2) |j| {
            const idx = i * 2 + j;
            std.debug.print("[{d:.1} + {d:.1}i] ", .{
                spectrum2d.data[idx].real,
                spectrum2d.data[idx].imag,
            });
        }
        std.debug.print("\n", .{});
    }
}
// Output:
// 2D FFT result:
// [10.0 + 0.0i] [-2.0 + 0.0i]
// [-4.0 + 0.0i] [0.0 + 0.0i]
```

## 2D IFFT

Inverse 2D FFT to recover original matrix:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2 };
    var data = [_]Complex(f32){
        Complex(f32){ .real = 1.0, .imag = 0.0 },
        Complex(f32){ .real = 2.0, .imag = 0.0 },
        Complex(f32){ .real = 3.0, .imag = 0.0 },
        Complex(f32){ .real = 4.0, .imag = 0.0 },
    };
    var matrix = num.core.NDArray(Complex(f32)).init(&shape, &data);
    
    var spectrum2d = try num.fft.fft2(allocator, &matrix);
    defer spectrum2d.deinit(allocator);
    
    // Inverse transform
    var reconstructed = try num.fft.ifft2(allocator, &spectrum2d);
    defer reconstructed.deinit(allocator);
    
    std.debug.print("Reconstructed 2D matrix:\n", .{});
    for (0..2) |i| {
        for (0..2) |j| {
            std.debug.print("{d:.1} ", .{reconstructed.data[i * 2 + j].real});
        }
        std.debug.print("\n", .{});
    }
}
// Output:
// Reconstructed 2D matrix:
// 1.0 2.0
// 3.0 4.0
```

## N-Dimensional FFT

Compute FFT along all axes of an N-dimensional array:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a 2x2x2 complex array
    const shape = [_]usize{ 2, 2, 2 };
    var data = [_]Complex(f32){
        Complex(f32){ .real = 1.0, .imag = 0.0 },
        Complex(f32){ .real = 2.0, .imag = 0.0 },
        Complex(f32){ .real = 3.0, .imag = 0.0 },
        Complex(f32){ .real = 4.0, .imag = 0.0 },
        Complex(f32){ .real = 5.0, .imag = 0.0 },
        Complex(f32){ .real = 6.0, .imag = 0.0 },
        Complex(f32){ .real = 7.0, .imag = 0.0 },
        Complex(f32){ .real = 8.0, .imag = 0.0 },
    };
    var tensor = num.core.NDArray(Complex(f32)).init(&shape, &data);
    
    // Compute N-D FFT
    var spectrum_nd = try num.fft.fftn(allocator, &tensor);
    defer spectrum_nd.deinit(allocator);
    
    std.debug.print("N-D FFT magnitude (first 4 values):\n", .{});
    for (0..4) |i| {
        const mag = @sqrt(spectrum_nd.data[i].real * spectrum_nd.data[i].real +
            spectrum_nd.data[i].imag * spectrum_nd.data[i].imag);
        std.debug.print("  {d:.2}\n", .{mag});
    }
}
// Output:
// N-D FFT magnitude (first 4 values):
//   36.00
//   8.00
//   8.00
//   0.00
```

## N-Dimensional IFFT

Inverse N-D FFT:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shape = [_]usize{ 2, 2, 2 };
    var data = [_]Complex(f32){
        Complex(f32){ .real = 1.0, .imag = 0.0 },
        Complex(f32){ .real = 2.0, .imag = 0.0 },
        Complex(f32){ .real = 3.0, .imag = 0.0 },
        Complex(f32){ .real = 4.0, .imag = 0.0 },
        Complex(f32){ .real = 5.0, .imag = 0.0 },
        Complex(f32){ .real = 6.0, .imag = 0.0 },
        Complex(f32){ .real = 7.0, .imag = 0.0 },
        Complex(f32){ .real = 8.0, .imag = 0.0 },
    };
    var tensor = num.core.NDArray(Complex(f32)).init(&shape, &data);
    
    var spectrum_nd = try num.fft.fftn(allocator, &tensor);
    defer spectrum_nd.deinit(allocator);
    
    // Inverse transform
    var reconstructed = try num.fft.ifftn(allocator, &spectrum_nd);
    defer reconstructed.deinit(allocator);
    
    std.debug.print("Reconstructed (first 4 values):\n", .{});
    for (0..4) |i| {
        std.debug.print("  {d:.1}\n", .{reconstructed.data[i].real});
    }
}
// Output:
// Reconstructed (first 4 values):
//   1.0
//   2.0
//   3.0
//   4.0
```

## Practical Example: Frequency Analysis

Analyze the frequency components of a composite signal:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a composite signal: 3Hz + 7Hz sine waves
    const n: usize = 128;
    const shape = [_]usize{n};
    var signal_data = try allocator.alloc(f32, n);
    defer allocator.free(signal_data);
    
    for (0..n) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        // Composite: sin(2π*3*t) + 0.5*sin(2π*7*t)
        signal_data[i] = @sin(2.0 * std.math.pi * 3.0 * t) +
            0.5 * @sin(2.0 * std.math.pi * 7.0 * t);
    }
    
    var signal = num.core.NDArray(f32).init(&shape, signal_data);
    
    // Compute FFT
    var spectrum = try num.fft.fft(allocator, &signal);
    defer spectrum.deinit(allocator);
    
    // Find dominant frequencies (peaks in magnitude)
    std.debug.print("Dominant frequency components:\n", .{});
    for (0..n / 2) |i| {
        const mag = @sqrt(spectrum.data[i].real * spectrum.data[i].real +
            spectrum.data[i].imag * spectrum.data[i].imag);
        if (mag > 30.0) {
            const freq = @as(f32, @floatFromInt(i));
            std.debug.print("  Frequency bin {d}: magnitude {d:.1}\n", .{ i, mag });
        }
    }
}
// Output:
// Dominant frequency components:
//   Frequency bin 3: magnitude 64.0
//   Frequency bin 7: magnitude 32.0
// (Correctly identifies 3Hz and 7Hz components)
```

## Practical Example: Image Filtering

Use 2D FFT to apply a low-pass filter to an image:

```zig
const std = @import("std");
const num = @import("num");
const Complex = num.complex.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a simple 8x8 "image" with a pattern
    const size: usize = 8;
    const shape = [_]usize{ size, size };
    var image_data = try allocator.alloc(Complex(f32), size * size);
    defer allocator.free(image_data);
    
    for (0..size) |i| {
        for (0..size) |j| {
            const val = if ((i + j) % 2 == 0) 1.0 else 0.0;
            image_data[i * size + j] = Complex(f32){ .real = val, .imag = 0.0 };
        }
    }
    
    var image = num.core.NDArray(Complex(f32)).init(&shape, image_data);
    
    // FFT
    var freq = try num.fft.fft2(allocator, &image);
    defer freq.deinit(allocator);
    
    // Apply low-pass filter (suppress high frequencies)
    const cutoff: usize = 3;
    for (0..size) |i| {
        for (0..size) |j| {
            if (i > cutoff or j > cutoff) {
                freq.data[i * size + j].real = 0.0;
                freq.data[i * size + j].imag = 0.0;
            }
        }
    }
    
    // Inverse FFT
    var filtered = try num.fft.ifft2(allocator, &freq);
    defer filtered.deinit(allocator);
    
    std.debug.print("Filtered image (low-pass):\n", .{});
    for (0..4) |i| {
        for (0..4) |j| {
            std.debug.print("{d:.2} ", .{filtered.data[i * size + j].real});
        }
        std.debug.print("\n", .{});
    }
}
// Output:
// Filtered image (low-pass):
// 0.50 0.50 0.50 0.50
// 0.50 0.50 0.50 0.50
// 0.50 0.50 0.50 0.50
// 0.50 0.50 0.50 0.50
// (Checkerboard pattern smoothed to uniform gray)
```