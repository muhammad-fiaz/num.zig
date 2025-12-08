const std = @import("std");
const Allocator = std.mem.Allocator;
const layers = @import("layers.zig");
const Layer = layers.Layer;
const autograd = @import("../autograd/tensor.zig");
const Tensor = autograd.Tensor;

/// A sequential container for neural network layers.
///
/// Logic: Chains layers together.
pub const Sequential = struct {
    layers: std.ArrayListUnmanaged(Layer),
    training: bool,

    /// Initializes a new Sequential model.
    pub fn init(allocator: Allocator) Sequential {
        _ = allocator;
        return .{
            .layers = .{},
            .training = true,
        };
    }

    /// Deinitializes the model and all its layers.
    pub fn deinit(self: *Sequential, allocator: Allocator) void {
        for (self.layers.items) |*layer| {
            layer.deinit(allocator);
        }
        self.layers.deinit(allocator);
    }

    /// Sets the model to training mode.
    pub fn train(self: *Sequential) void {
        self.training = true;
        for (self.layers.items) |*layer| {
            layer.train();
        }
        std.debug.print("Model set to training mode.\n", .{});
    }

    /// Sets the model to evaluation mode.
    pub fn eval(self: *Sequential) void {
        self.training = false;
        for (self.layers.items) |*layer| {
            layer.eval();
        }
        std.debug.print("Model set to evaluation mode.\n", .{});
    }

    /// Adds a layer to the model.
    pub fn add(self: *Sequential, allocator: Allocator, layer: Layer) !void {
        try self.layers.append(allocator, layer);
    }

    /// Performs the forward pass through all layers.
    pub fn forward(self: *Sequential, allocator: Allocator, input: *Tensor(f32)) !*Tensor(f32) {
        var current = input;
        for (self.layers.items) |*layer| {
            current = try layer.forward(allocator, current);
        }
        return current;
    }

    /// Trains the model for a fixed number of epochs.
    ///
    /// Arguments:
    ///     allocator: Allocator for intermediate tensors.
    ///     x_train: Training data features.
    ///     y_train: Training data targets.
    ///     optimizer: Optimizer instance (must have `update` method).
    ///     loss_fn: Loss function (must take allocator, pred, target and return !*Tensor).
    ///     epochs: Number of epochs to train.
    pub fn fit(
        self: *Sequential,
        allocator: Allocator,
        x_train: *Tensor(f32),
        y_train: *Tensor(f32),
        optimizer: anytype,
        loss_fn: anytype,
        epochs: usize,
    ) !void {
        self.train();
        std.debug.print("Starting training for {} epochs...\n", .{epochs});

        for (0..epochs) |epoch| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const frame_alloc = arena.allocator();

            // Forward
            const out = try self.forward(frame_alloc, x_train);

            // Loss
            const loss = try loss_fn(frame_alloc, out, y_train);

            if (epoch % 10 == 0 or epoch == epochs - 1) {
                std.debug.print("Epoch {}: Loss = {d:.6}\n", .{ epoch, loss.data.data[0] });
            }

            // Zero Gradients
            const params = try self.parameters(frame_alloc);
            for (params.items) |param| {
                if (param.grad) |*g| {
                    g.fill(0);
                }
            }

            // Backward
            try loss.backward(frame_alloc);

            // Update
            for (params.items) |param| {
                optimizer.update(param);
            }
        }
        std.debug.print("Training completed.\n", .{});
    }

    /// Saves the model to a file.
    ///
    /// Format:
    /// - Magic: "NUMZIG_MODEL" (12 bytes)
    /// - Version: u8 (1 byte)
    /// - Num Layers: u64 (8 bytes)
    /// - Layers...
    pub fn save(self: Sequential, allocator: Allocator, path: []const u8) !void {
        std.debug.print("Saving model to '{s}'...\n", .{path});
        const file = std.fs.cwd().createFile(path, .{}) catch |err| {
            std.debug.print("Error creating file '{s}': {}\n", .{ path, err });
            return err;
        };
        defer file.close();
        // Use file directly as writer since we use writeAll
        const writer = file;

        try writer.writeAll("NUMZIG_MODEL");

        var buf: [8]u8 = undefined;

        // Version
        buf[0] = 1;
        try writer.writeAll(buf[0..1]);

        // Num Layers
        std.mem.writeInt(u64, &buf, self.layers.items.len, .little);
        try writer.writeAll(&buf);

        for (self.layers.items) |layer| {
            try layer.save(allocator, writer);
        }
        std.debug.print("Model saved successfully.\n", .{});
    }

    /// Loads a model from a file.
    pub fn load(allocator: Allocator, path: []const u8) !Sequential {
        std.debug.print("Loading model from '{s}'...\n", .{path});
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            std.debug.print("Error opening file '{s}': {}\n", .{ path, err });
            return err;
        };
        defer file.close();
        const reader = file;

        var magic: [12]u8 = undefined;
        if ((try reader.readAll(&magic)) != magic.len) return error.EndOfStream;
        if (!std.mem.eql(u8, &magic, "NUMZIG_MODEL")) {
            std.debug.print("Error: Invalid model format.\n", .{});
            return error.InvalidFormat;
        }

        var buf: [8]u8 = undefined;

        // Version
        if ((try reader.readAll(buf[0..1])) != 1) return error.EndOfStream;
        if (buf[0] != 1) return error.UnsupportedVersion;

        // Num Layers
        if ((try reader.readAll(&buf)) != 8) return error.EndOfStream;
        const num_layers = std.mem.readInt(u64, &buf, .little);

        var model = Sequential.init(allocator);
        errdefer model.deinit(allocator);

        for (0..num_layers) |_| {
            const layer = try Layer.load(allocator, reader);
            try model.add(allocator, layer);
        }

        std.debug.print("Model loaded successfully ({} layers).\n", .{num_layers});
        return model;
    }

    /// Returns a list of all trainable parameters in the model.
    pub fn parameters(self: *Sequential, allocator: Allocator) !std.ArrayListUnmanaged(*Tensor(f32)) {
        var list = std.ArrayListUnmanaged(*Tensor(f32)){};
        for (self.layers.items) |*layer| {
            var layer_params = try layer.parameters(allocator);
            defer layer_params.deinit(allocator);
            try list.appendSlice(allocator, layer_params.items);
        }
        return list;
    }
};

test "Sequential save/load" {
    const allocator = std.testing.allocator;
    const Dense = layers.Dense;

    var model = Sequential.init(allocator);
    defer model.deinit(allocator);

    var l1 = try Dense.init(allocator, 2, 3, .XavierUniform);
    // Manually set weights for deterministic test
    l1.weights.data.data[0] = 0.5;
    try model.add(allocator, Layer{ .Dense = l1 });
    try model.add(allocator, Layer{ .ReLU = {} });

    try model.save(allocator, "test_model.bin");
    defer std.fs.cwd().deleteFile("test_model.bin") catch {};

    var loaded = try Sequential.load(allocator, "test_model.bin");
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(loaded.layers.items.len, 2);

    switch (loaded.layers.items[0]) {
        .Dense => |l| try std.testing.expectEqual(l.weights.data.data[0], 0.5),
        else => try std.testing.expect(false),
    }
}

test "Sequential training and save/load" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const Dense = layers.Dense;
    const NDArray = @import("../core.zig").NDArray;
    const optim = @import("optim.zig");

    // 1. Create Model
    var model = Sequential.init(arena_alloc);

    try model.add(arena_alloc, Layer{ .Dense = try Dense.init(arena_alloc, 2, 4, .XavierUniform) });
    try model.add(arena_alloc, Layer{ .Tanh = {} });
    try model.add(arena_alloc, Layer{ .Dense = try Dense.init(arena_alloc, 4, 1, .XavierUniform) });
    try model.add(arena_alloc, Layer{ .Sigmoid = {} });

    // 2. Train (One step)
    var x_data = try NDArray(f32).init(arena_alloc, &.{ 1, 2 });
    x_data.fill(1.0);
    const x = try Tensor(f32).init(arena_alloc, x_data, false);

    var y_data = try NDArray(f32).init(arena_alloc, &.{ 1, 1 });
    y_data.fill(0.0);
    const y = try Tensor(f32).init(arena_alloc, y_data, false);

    // Forward
    const out = try model.forward(arena_alloc, x);
    const loss = try out.mse_loss(arena_alloc, y);

    // Backward
    try loss.backward(arena_alloc);

    // Update
    var optimizer = optim.SGD.init(0.1);
    const params = try model.parameters(arena_alloc);

    for (params.items) |param| {
        optimizer.update(param);
    }

    // 3. Save
    try model.save(arena_alloc, "trained_model.bin");
    defer std.fs.cwd().deleteFile("trained_model.bin") catch {};

    // 4. Load
    var loaded = try Sequential.load(arena_alloc, "trained_model.bin");

    // 5. Verify Inference
    const out_orig = try model.forward(arena_alloc, x);
    const out_loaded = try loaded.forward(arena_alloc, x);

    try std.testing.expectApproxEqAbs(out_orig.data.data[0], out_loaded.data.data[0], 1e-6);
}

test "Sequential fit" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const Dense = layers.Dense;
    const NDArray = @import("../core.zig").NDArray;
    const optim = @import("optim.zig");

    var model = Sequential.init(arena_alloc);
    try model.add(arena_alloc, Layer{ .Dense = try Dense.init(arena_alloc, 2, 1, .XavierUniform) });

    var x_data = try NDArray(f32).init(arena_alloc, &.{ 1, 2 });
    x_data.fill(1.0);
    const x = try Tensor(f32).init(arena_alloc, x_data, false);

    var y_data = try NDArray(f32).init(arena_alloc, &.{ 1, 1 });
    y_data.fill(0.0);
    const y = try Tensor(f32).init(arena_alloc, y_data, false);

    const optimizer = optim.SGD.init(0.01);

    const LossWrapper = struct {
        pub fn mse(alloc: Allocator, pred: *Tensor(f32), target: *Tensor(f32)) !*Tensor(f32) {
            return pred.mse_loss(alloc, target);
        }
    };

    try model.fit(arena_alloc, x, y, optimizer, LossWrapper.mse, 2);
}
