# Operations Guide

NumZig supports a wide range of mathematical operations.

## Elementwise Operations

These operations apply a function to each element of the array(s). They support **broadcasting**, meaning arrays of different shapes can be combined if their dimensions are compatible.

- `add(a, b)`
- `sub(a, b)`
- `mul(a, b)`
- `div(a, b)`
- `pow(a, b)`
- `exp(a)`
- `log(a)`
- `sqrt(a)`
- `sin(a)`, `cos(a)`, `tan(a)`
- `arcsin(a)`, `arccos(a)`, `arctan(a)`
- `logical_and(a, b)`, `logical_or(a, b)`, `logical_xor(a, b)`, `logical_not(a)`
- `bitwise_and(a, b)`, `bitwise_or(a, b)`, `bitwise_xor(a, b)`
- `equal(a, b)`, `greater(a, b)`, `less(a, b)`

```zig
var c = try num.elementwise.add(allocator, f32, a, b);
```

## Linear Algebra

Located in `num.linalg`.

- `dot(a, b)`: Dot product of two 1D arrays.
- `matmul(a, b)`: Matrix multiplication of two 2D arrays.
- `solve(a, b)`: Solve linear system Ax = B.
- `inverse(a)`: Multiplicative inverse of a matrix.
- `determinant(a)`: Determinant of a matrix.
- `trace(a)`: Sum of diagonal elements.
- `norm(a)`: Matrix or vector norm.
- `cholesky(a)`: Cholesky decomposition.


## Statistics

Located in `num.stats`.

- `sum(a)`, `sumAxis(a, axis)`
- `mean(a)`, `meanAxis(a, axis)`
- `stdDev(a)`, `variance(a)`
- `min(a)`, `max(a)`
- `argmin(a)`, `argmax(a)`
- `median(a)`, `percentile(a, q)`
- `histogram(a, bins, min, max)`
- `bincount(a)`

## Sorting

Located in `num.sort`.

- `sort(a, axis)`: Return a sorted copy.
- `argsort(a, axis)`: Return indices that would sort the array.

## Random

Located in `num.random`.

- `uniform(shape)`
- `normal(shape, mean, std)`
- `randint(shape, low, high)`
- `shuffle(arr)`
