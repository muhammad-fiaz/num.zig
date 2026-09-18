---
layout: home
title: "num.zig | High-Performance Numerical Computing for Zig"
description: "A fast, production-ready, high-performance numerical computing and N-dimensional array library for Zig with SIMD acceleration, linear algebra, FFT, statistics, multi-threaded CPU parallel execution, and portable NZIG v1.0 binary serialization."

hero:
  name: "num.zig"
  text: "High-Performance Numerical Computing for Zig"
  tagline: "Production-ready, pure Zig N-dimensional arrays with SIMD math, linear algebra, multi-threaded CPU parallelism, and portable NZIG v1.0 serialization."
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /api/
    - theme: alt
      text: View on GitHub
      link: https://github.com/muhammad-fiaz/num.zig

features:
  - title: N-Dimensional Arrays & SBO
    details: Small Buffer Optimization (SBO) up to 8D inline with zero-copy views for slicing, reshaping, transposing, and broadcasting.
  - title: Vectorized Math & SIMD
    details: Auto-vectorized elementwise arithmetic, trigonometry, special functions (erf, hypot, atan2), and boolean logic.
  - title: Linear Algebra & Decompositions
    details: Blocked cache-friendly GEMM, LU, QR, Cholesky, SVD, Eigenvalues/vectors, Moore-Penrose pseudoinverse, and matrix powers.
  - title: Multi-Threaded CPU Parallelism
    details: Deterministic data-parallel engine with inline configuration, automatic core detection, sequential fallback, and nested parallelism protection.
  - title: Fast Fourier Transform & Sparse
    details: 1D/2D Radix-2 Cooley-Tukey FFT/IFFT with complex spectra, plus CSR/CSC sparse matrices with transpose support and iterative Conjugate Gradient and GMRES solvers.
  - title: Native NZIG v1.0 Serialization
    details: Portable, 64-byte aligned, deterministic binary array file format independent of host CPU ABI and endianness.
---
