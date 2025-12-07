const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the num module
    const num_module = b.createModule(.{
        .root_source_file = b.path("src/num.zig"),
    });

    // Expose the module for external projects that depend on this package.
    // This allows users to do: `const num = @import("num");` in their code
    // after adding num as a dependency and calling `dep.module("num")` in their build.zig
    _ = b.addModule("num", .{
        .root_source_file = b.path("src/num.zig"),
    });

    // Build examples
    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "basics", .path = "examples/01_basics.zig" },
        .{ .name = "manipulation", .path = "examples/02_manipulation.zig" },
        .{ .name = "math", .path = "examples/03_math.zig" },
        .{ .name = "linalg", .path = "examples/04_linalg.zig" },
        .{ .name = "random", .path = "examples/05_random.zig" },
        .{ .name = "ml", .path = "examples/06_ml.zig" },
        .{ .name = "fft", .path = "examples/07_fft.zig" },
        .{ .name = "indexing", .path = "examples/08_indexing.zig" },
        .{ .name = "signal_poly", .path = "examples/09_signal_poly.zig" },
        .{ .name = "setops", .path = "examples/10_setops.zig" },
        .{ .name = "complex", .path = "examples/11_complex.zig" },
        .{ .name = "advanced_manipulation", .path = "examples/12_advanced_manipulation.zig" },
        .{ .name = "algorithms", .path = "examples/13_algorithms.zig" },
        .{ .name = "csv_linalg_masking", .path = "examples/14_csv_linalg_masking.zig" },
        .{ .name = "comprehensive", .path = "examples/15_comprehensive.zig" },
    };

    inline for (examples) |example| {
        const exe = b.addExecutable(.{
            .name = example.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(example.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("num", num_module);

        const install_exe = b.addInstallArtifact(exe, .{});
        const example_step = b.step("example-" ++ example.name, "Build " ++ example.name ++ " example");
        example_step.dependOn(&install_exe.step);

        // Add run step for each example
        const run_exe = b.addRunArtifact(exe);
        run_exe.step.dependOn(&install_exe.step);
        const run_step = b.step("run-" ++ example.name, "Run " ++ example.name ++ " example");
        run_step.dependOn(&run_exe.step);
    }

    // Benchmark Step
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/benchmark.zig"),
            .target = target,
            .optimize = .ReleaseFast, // Benchmarks should be release
        }),
    });
    bench_exe.root_module.addImport("num", num_module);

    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("benchmark", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);

    // Unit tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/num.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Install step for library
    const lib = b.addLibrary(.{
        .name = "num",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/num.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);
}
