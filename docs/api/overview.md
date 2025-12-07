# API Reference

## Core (`num.core`)

| Function | Description |
| :--- | :--- |
| `NDArray(T)` | Main array struct. |
| `init(allocator, shape)` | Create uninitialized array. |
| `zeros(allocator, shape)` | Create array of zeros. |
| `ones(allocator, shape)` | Create array of ones. |
| `full(allocator, shape, val)` | Create array filled with value. |
| `arange(allocator, start, stop, step)` | Create range. |
| `linspace(allocator, start, stop, num)` | Create linearly spaced values. |
| `eye(allocator, n)` | Create identity matrix. |
| `reshape(shape)` | Return reshaped view. |
| `transpose()` | Return transposed view (reverse axes). |
| `permute(axes)` | Return permuted view. |
| `flatten()` | Return flattened copy. |
| `squeeze()` | Remove 1-dims. |
| `stack(allocator, arrays, axis)` | Stack arrays. |
| `concatenate(allocator, arrays, axis)` | Concatenate arrays. |
| `broadcastTo(shape)` | Broadcast to shape. |

## Manipulation (`num.manipulation`)

| Function | Description |
| :--- | :--- |
| `vstack(arrays)` | Stack vertically. |
| `hstack(arrays)` | Stack horizontally. |
| `dstack(arrays)` | Stack depth-wise. |
| `tile(arr, reps)` | Tile array. |
| `repeat(arr, reps, axis)` | Repeat elements. |
| `moveaxis(arr, src, dst)` | Move axes. |
| `ravel(arr)` | Flatten to contiguous. |

## Operations (`num.elementwise`)

| Function | Description |
| :--- | :--- |
| `add(a, b)` | Elementwise addition. |
| `sub(a, b)` | Elementwise subtraction. |
| `mul(a, b)` | Elementwise multiplication. |
| `div(a, b)` | Elementwise division. |
| `pow(a, b)` | Elementwise power. |
| `exp(a)` | Elementwise exponential. |
| `log(a)` | Elementwise natural logarithm. |
| `sqrt(a)` | Elementwise square root. |
| `sin(a)` | Elementwise sine. |
| `cos(a)` | Elementwise cosine. |
| `tan(a)` | Elementwise tangent. |
| `arcsin(a)` | Elementwise arcsin. |
| `arccos(a)` | Elementwise arccos. |
| `arctan(a)` | Elementwise arctan. |
| `logical_and(a, b)` | Logical AND. |
| `logical_or(a, b)` | Logical OR. |
| `bitwise_and(a, b)` | Bitwise AND. |
| `equal(a, b)` | Elementwise equality. |

## Linear Algebra (`num.linalg`)

| Function | Description |
| :--- | :--- |
| `dot(a, b)` | Dot product. |
| `matmul(a, b)` | Matrix multiplication. |
| `solve(a, b)` | Solve linear system. |
| `inverse(a)` | Matrix inverse. |
| `determinant(a)` | Matrix determinant. |
| `trace(a)` | Matrix trace. |
| `norm(a)` | Matrix/Vector norm. |
| `cholesky(a)` | Cholesky decomposition. |

## Indexing (`num.indexing`)

| Function | Description |
| :--- | :--- |
| `booleanMask(arr, mask)` | Select with mask. |
| `where(cond, x, y)` | Conditional selection. |

## Sorting (`num.sort`)

| Function | Description |
| :--- | :--- |
| `sort(arr, axis)` | Sort array. |
| `argsort(arr, axis)` | Indices to sort. |
| `nonzero(arr)` | Indices of non-zero elements. |

## Statistics (`num.stats`)


| Function | Description |
| :--- | :--- |
| `sum(a)`, `sumAxis` | Sum of elements. |
| `mean(a)`, `meanAxis` | Mean of elements. |
| `stdDev(a)` | Standard deviation. |
| `variance(a)` | Variance. |
| `min(a)`, `argmin(a)` | Minimum value/index. |
| `max(a)`, `argmax(a)` | Maximum value/index. |
| `median(a)` | Median value. |
| `percentile(a, q)` | q-th percentile. |
| `histogram(a, ...)` | Histogram. |

## Complex Numbers (`num.complex`)

| Function | Description |
| :--- | :--- |
| `real(a)` | Real part. |
| `imag(a)` | Imaginary part. |
| `conj(a)` | Complex conjugate. |
| `abs(a)` | Magnitude. |

## FFT (`num.fft`)

| Function | Description |
| :--- | :--- |
| `fft(a)` | 1D FFT. |
| `ifft(a)` | 1D Inverse FFT. |
| `fft2(a)` | 2D FFT. |
| `fftn(a)` | N-D FFT. |

## Sparse (`num.sparse`)

| Function | Description |
| :--- | :--- |
| `CSRMatrix` | Compressed Sparse Row matrix. |
| `fromDense(arr)` | Convert dense to sparse. |
| `toDense()` | Convert sparse to dense. |

## Autograd (`num.autograd`)

| Function | Description |
| :--- | :--- |
| `Tensor` | Tensor with autograd support. |
| `backward()` | Compute gradients. |

## DataFrame (`num.dataframe`)

| Function | Description |
| :--- | :--- |
| `DataFrame` | Tabular data structure. |
| `Series` | 1D labeled array. |

## Optimization (`num.optimize`)

| Function | Description |
| :--- | :--- |
| `GradientDescent` | Gradient Descent optimizer. |

## Interpolation (`num.interpolate`)

| Function | Description |
| :--- | :--- |
| `interp1d(x, y, xi)` | 1D linear interpolation. |

## Input / Output (`num.io`)

| Function | Description |
| :--- | :--- |
| `save(arr, path)` | Save array to binary file. |
| `load(path)` | Load array from binary file. |
| `saveCSV(arr, path)` | Save array to CSV. |
| `loadCSV(path)` | Load array from CSV. |
| `bincount(a)` | Count occurrences. |

## Random (`num.random`)

| Function | Description |
| :--- | :--- |
| `Random.init(seed)` | Initialize RNG. |
| `uniform(shape)` | Uniform [0, 1). |
| `normal(shape, mean, std)` | Normal distribution. |
| `randint(shape, low, high)` | Uniform integers. |
| `shuffle(arr)` | Shuffle array in-place. |

## Sorting (`num.sort`)

| Function | Description |
| :--- | :--- |
| `sort(a, axis)` | Return sorted copy. |
| `argsort(a, axis)` | Return sort indices. |
| `partition(a, kth, axis)` | Partition array. |
| `argpartition(a, kth, axis)` | Partition indices. |
| `nonzero(a)` | Indices of non-zero elements. |
| `flatnonzero(a)` | Indices of non-zero elements in flattened array. |

## Polynomials (`num.poly`)

| Function | Description |
| :--- | :--- |
| `polyval(p, x)` | Evaluate polynomial. |
| `polyadd(p1, p2)` | Add polynomials. |
| `polysub(p1, p2)` | Subtract polynomials. |
| `polymul(p1, p2)` | Multiply polynomials. |
| `roots(p)` | Find roots. |
| `polyder(p, m)` | Derivative. |
| `polyint(p, m, k)` | Integral. |

## Signal Processing (`num.signal`)

| Function | Description |
| :--- | :--- |
| `convolve(a, v, mode)` | Convolution. |
| `correlate(a, v, mode)` | Cross-correlation. |

## Set Operations (`num.setops`)

| Function | Description |
| :--- | :--- |
| `unique(a)` | Unique elements. |
| `in1d(ar1, ar2)` | Membership test. |
| `intersect1d(ar1, ar2)` | Intersection. |
| `union1d(ar1, ar2)` | Union. |
| `setdiff1d(ar1, ar2)` | Set difference. |
| `setxor1d(ar1, ar2)` | Set XOR. |

## FFT (`num.fft`)

| Function | Description |
| :--- | :--- |
| `fft(a)` | 1D Fast Fourier Transform. |
| `ifft(a)` | 1D Inverse FFT. |

## Machine Learning (`num.ml`)

| Module | Description |
| :--- | :--- |
| `layers` | `Dense`, `Dropout`. |
| `activations` | `relu`, `sigmoid`, `tanh`, `softmax`, etc. |
| `loss` | `mse`, `categoricalCrossEntropy`, etc. |
| `optim` | `SGD`, `Momentum`, `Adam`. |
