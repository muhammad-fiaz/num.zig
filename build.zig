const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the num module
    const num_module = b.createModule(.{
        .root_source_file = b.path("src/num.zig"),
    });

    // Expose the module for external projects that depend on this package.
    _ = b.addModule("num", .{
        .root_source_file = b.path("src/num.zig"),
    });

    const examples = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "array_creation", .path = "examples/array_creation.zig" },
        .{ .name = "elementwise", .path = "examples/elementwise.zig" },
        .{ .name = "broadcasting", .path = "examples/broadcasting.zig" },
        .{ .name = "slicing", .path = "examples/slicing.zig" },
        .{ .name = "reshape", .path = "examples/reshape.zig" },
        .{ .name = "transpose", .path = "examples/transpose.zig" },
        .{ .name = "concatenation", .path = "examples/concatenation.zig" },
        .{ .name = "reductions", .path = "examples/reductions.zig" },
        .{ .name = "sorting", .path = "examples/sorting.zig" },
        .{ .name = "searching", .path = "examples/searching.zig" },
        .{ .name = "random_generation", .path = "examples/random_generation.zig" },
        .{ .name = "statistics", .path = "examples/statistics.zig" },
        .{ .name = "correlation", .path = "examples/correlation.zig" },
        .{ .name = "matmul", .path = "examples/matmul.zig" },
        .{ .name = "norms", .path = "examples/norms.zig" },
        .{ .name = "solve", .path = "examples/solve.zig" },
        .{ .name = "decompositions", .path = "examples/decompositions.zig" },
        .{ .name = "polynomials", .path = "examples/polynomials.zig" },
        .{ .name = "serialization", .path = "examples/serialization.zig" },
        .{ .name = "eigenvalues", .path = "examples/eigenvalues.zig" },
        .{ .name = "svd", .path = "examples/svd.zig" },
        .{ .name = "fft", .path = "examples/fft.zig" },
        .{ .name = "sparse_matrix", .path = "examples/sparse_matrix.zig" },
        .{ .name = "parallel_basic", .path = "examples/parallel_basic.zig" },
        .{ .name = "parallel_config", .path = "examples/parallel_config.zig" },
        .{ .name = "parallel_large_array", .path = "examples/parallel_large_array.zig" },
        .{ .name = "parallel_f32", .path = "examples/parallel_f32.zig" },
        .{ .name = "parallel_f64", .path = "examples/parallel_f64.zig" },
        .{ .name = "parallel_non_contiguous", .path = "examples/parallel_non_contiguous.zig" },
        .{ .name = "parallel_reduction", .path = "examples/parallel_reduction.zig" },
    };

    // Build all examples step
    const build_all_examples = b.step("build-all-examples", "Build all example executables");
    const examples_step = b.step("examples", "Build all user-facing examples");

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
        build_all_examples.dependOn(&install_exe.step);
        examples_step.dependOn(&install_exe.step);

        const example_step = b.step("example-" ++ example.name, "Build " ++ example.name ++ " example");
        example_step.dependOn(&install_exe.step);

        // Add individual run step for each example
        const run_exe = b.addRunArtifact(exe);
        run_exe.step.dependOn(&install_exe.step);
        const run_step = b.step("run-" ++ example.name, "Run " ++ example.name ++ " example");
        run_step.dependOn(&run_exe.step);
    }

    // Run all examples sequentially step
    const run_all_examples = b.step("run-all-examples", "Run all examples sequentially");
    var prev_step: ?*std.Build.Step = null;

    inline for (examples) |example| {
        const exe = b.addExecutable(.{
            .name = "run-all-" ++ example.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(example.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("num", num_module);

        const install_exe = b.addInstallArtifact(exe, .{});
        if (prev_step) |p| {
            exe.step.dependOn(p);
            install_exe.step.dependOn(p);
        }
        const run_exe = b.addRunArtifact(exe);
        run_exe.step.dependOn(&install_exe.step);

        prev_step = &run_exe.step;
    }

    if (prev_step) |last| {
        run_all_examples.dependOn(last);
    }

    // Benchmark Step
    const bench_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/benchmark.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    bench_exe.root_module.addImport("num", num_module);

    const install_bench = b.addInstallArtifact(bench_exe, .{});
    const run_bench = b.addRunArtifact(bench_exe);
    run_bench.step.dependOn(&install_bench.step);

    const bench_step = b.step("benchmark", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);
    const bench_alias = b.step("bench", "Run benchmarks (alias)");
    bench_alias.dependOn(&run_bench.step);

    // Unit tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/num.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_tests = b.addRunArtifact(tests);
    run_tests.has_side_effects = true;
    const test_step = b.step("test", "Run unit tests");

    // Only run tests when target matches host; otherwise compile test artifact only.
    if (target.result.os.tag == builtin.os.tag and target.result.cpu.arch == builtin.cpu.arch) {
        test_step.dependOn(&run_tests.step);
    } else {
        const install_tests = b.addInstallArtifact(tests, .{});
        test_step.dependOn(&install_tests.step);
    }

    const test_check_step = b.step("test-check", "Compile unit tests without running (useful for cross-compilation)");
    test_check_step.dependOn(&tests.step);

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

    // Docs step
    const docs_step = b.step("docs", "Generate library documentation");
    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&install_docs.step);

    // Test-all step: run tests, benchmark, and all runnable examples
    const test_all_step = b.step("test-all", "Run tests, benchmarks, and all runnable examples");
    test_all_step.dependOn(test_step);
    test_all_step.dependOn(bench_step);
    test_all_step.dependOn(run_all_examples);
}
