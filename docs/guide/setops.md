# Set Operations

`num.zig` supports set operations on 1D arrays via `num.setops`.

## Unique Elements

Find unique elements in an array.

```zig
const num = @import("num");

var a = ...; // [1, 2, 3, 2, 1]
var u = try num.setops.unique(allocator, i32, a);
// u is [1, 2, 3]
```

## Membership Test

Test if elements of one array are present in another.

```zig
var mask = try num.setops.in1d(allocator, i32, a, b);
```

## Set Logic

- `intersect1d`: Intersection
- `union1d`: Union
- `setdiff1d`: Difference
- `setxor1d`: Exclusive OR

```zig
var i = try num.setops.intersect1d(allocator, i32, a, b);
```
