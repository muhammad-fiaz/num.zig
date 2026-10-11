# Contributing to Num.Zig

Thank you for your interest in contributing to Num.Zig! We welcome contributions from the community to help make this library better.

> **Compiler requirement:** num.zig v0.0.4 targets **Zig 0.17.0** as its minimum supported compiler version. Library v0.0.3 remains the Zig 0.16.0 compatibility line.

## How to Contribute

1.  **Fork the Repository**: Click the "Fork" button on the top right of the repository page.
2.  **Clone your Fork**:
    ```bash
    git clone https://github.com/your-username/num.zig.git
    cd num.zig
    ```
3.  **Create a Branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```
4.  **Make Changes**: Implement your feature or fix a bug.
5.  **Run Tests**: Ensure all tests pass (requires Zig 0.17.0).
    ```bash
    zig build test
    ```
6.  **Run Examples**: Ensure examples run correctly.
    ```bash
    zig build run-all-examples
    ```
    **Run Benchmarks**:
    ```bash
    zig build bench
    ```
    **Check Formatting**:
    ```bash
    zig fmt --check src/ bench/ examples/ build.zig
    ```
7.  **Commit Changes**:
    ```bash
    git commit -m "Description of your changes"
    ```
8.  **Push to GitHub**:
    ```bash
    git push origin feature/your-feature-name
    ```
9.  **Open a Pull Request**: Go to the original repository and open a Pull Request.

## Coding Style

- Follow standard Zig coding conventions.
- Use `zig fmt` to format your code.
- Add comments and documentation for public functions.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub repository. Provide as much detail as possible, including reproduction steps for bugs.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
