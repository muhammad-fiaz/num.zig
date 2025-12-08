const std = @import("std");
const core = @import("../core.zig");
const linalg = @import("../linalg.zig");
const random = @import("../random.zig");
const autograd = @import("../autograd/tensor.zig");
const io = @import("../io.zig");
const NDArray = core.NDArray;
const Tensor = autograd.Tensor;
const Allocator = std.mem.Allocator;

/// Initialization methods for neural network weights.
pub const InitMethod = enum {
    RandomUniform,
    XavierUniform,
    HeNormal,
};

/// Represents a fully connected (dense) neural network layer.
///
/// A dense layer performs a linear transformation on the input data.
///
/// Logic:
/// output = input @ weights + bias
pub const Dense = struct {
    weights: *Tensor(f32),
    bias: *Tensor(f32),
    training: bool,

    /// Initializes a new Dense layer with the specified input and output dimensions.
    ///
    /// Arguments:
    ///     allocator: The allocator to use for initializing weights and bias.
    ///     input_dim: The number of input features.
    ///     output_dim: The number of output features.
    ///     init_method: The initialization method for weights (default: .XavierUniform).
    ///
    /// Returns:
    ///     A new Dense layer instance.
    ///
    /// Example:
    /// ```zig
    /// var layer = try Dense.init(allocator, 784, 128, .XavierUniform);
    /// defer layer.deinit(allocator);
    /// ```
    pub fn init(allocator: Allocator, input_dim: usize, output_dim: usize, init_method: InitMethod) !Dense {
        const w_data = try NDArray(f32).init(allocator, &.{ input_dim, output_dim });

        var prng = std.Random.DefaultPrng.init(0);
        const rand = prng.random();

        switch (init_method) {
            .RandomUniform => {
                for (w_data.data) |*val| val.* = rand.float(f32) * 0.02 - 0.01;
            },
            .XavierUniform => {
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_dim + output_dim)));
                for (w_data.data) |*val| val.* = rand.float(f32) * (2.0 * limit) - limit;
            },
            .HeNormal => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_dim)));
                for (w_data.data) |*val| val.* = rand.floatNorm(f32) * std_dev;
            },
        }

        const w = try Tensor(f32).init(allocator, w_data, true);
        errdefer w.deinit(allocator);

        const b_data = try NDArray(f32).zeros(allocator, &.{output_dim});
        const b = try Tensor(f32).init(allocator, b_data, true);

        return Dense{ .weights = w, .bias = b, .training = true };
    }

    /// Deinitializes the layer, freeing associated memory.
    pub fn deinit(self: *Dense, allocator: Allocator) void {
        self.weights.deinit(allocator);
        self.bias.deinit(allocator);
    }

    /// Sets the layer to training mode.
    pub fn train(self: *Dense) void {
        self.training = true;
    }

    /// Sets the layer to evaluation mode.
    pub fn eval(self: *Dense) void {
        self.training = false;
    }

    /// Performs the forward pass of the layer.
    ///
    /// Computes the output of the dense layer.
    ///
    /// Logic:
    /// z = matmul(input, weights) + bias
    ///
    /// Arguments:
    ///     self: The Dense layer instance.
    ///     allocator: The allocator to use for the output array.
    ///     input: The input array of shape (batch_size, input_dim).
    ///
    /// Returns:
    ///     The output of the layer of shape (batch_size, output_dim).
    ///
    /// Example:
    /// ```zig
    /// var output = try layer.forward(allocator, &input);
    /// defer output.deinit(allocator);
    /// ```
    pub fn forward(self: *Dense, allocator: Allocator, input: *Tensor(f32)) !*Tensor(f32) {
        const z = try input.matmul(allocator, self.weights);
        return z.add(allocator, self.bias);
    }

    /// Saves the layer weights and biases to a writer.
    pub fn save(self: Dense, allocator: Allocator, writer: anytype) !void {
        try io.writeArray(allocator, f32, self.weights.data, writer);
        try io.writeArray(allocator, f32, self.bias.data, writer);
    }

    /// Loads a Dense layer from a reader.
    pub fn load(allocator: Allocator, reader: anytype) !Dense {
        const w_data = try io.readArray(allocator, f32, reader);
        const b_data = try io.readArray(allocator, f32, reader);

        const w = try Tensor(f32).init(allocator, w_data, true);
        const b = try Tensor(f32).init(allocator, b_data, true);

        return Dense{ .weights = w, .bias = b, .training = true };
    }
};

pub const LayerType = enum {
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
};

pub const Layer = union(LayerType) {
    Dense: Dense,
    ReLU: void,
    Sigmoid: void,
    Tanh: void,
    Softmax: void,

    pub fn forward(self: *Layer, allocator: Allocator, input: *Tensor(f32)) !*Tensor(f32) {
        switch (self.*) {
            .Dense => |*l| return l.forward(allocator, input),
            .ReLU => return input.relu(allocator),
            .Sigmoid => return input.sigmoid(allocator),
            .Tanh => return input.tanh(allocator),
            .Softmax => return input.softmax(allocator),
        }
    }

    pub fn train(self: *Layer) void {
        switch (self.*) {
            .Dense => |*l| l.train(),
            else => {},
        }
    }

    pub fn eval(self: *Layer) void {
        switch (self.*) {
            .Dense => |*l| l.eval(),
            else => {},
        }
    }

    pub fn deinit(self: *Layer, allocator: Allocator) void {
        switch (self.*) {
            .Dense => |*l| l.deinit(allocator),
            else => {},
        }
    }

    pub fn save(self: Layer, allocator: Allocator, writer: anytype) !void {
        // Write layer type ID
        const type_id = @intFromEnum(self);
        var buf: [1]u8 = undefined;
        std.mem.writeInt(u8, &buf, @intCast(type_id), .little);
        try writer.writeAll(&buf);

        switch (self) {
            .Dense => |l| try l.save(allocator, writer),
            else => {}, // No state for activations
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Layer {
        var buf: [1]u8 = undefined;
        if ((try reader.readAll(&buf)) != buf.len) return error.EndOfStream;
        const type_id = std.mem.readInt(u8, &buf, .little);
        const layer_type: LayerType = @enumFromInt(type_id);

        switch (layer_type) {
            .Dense => {
                const l = try Dense.load(allocator, reader);
                return Layer{ .Dense = l };
            },
            .ReLU => return Layer{ .ReLU = {} },
            .Sigmoid => return Layer{ .Sigmoid = {} },
            .Tanh => return Layer{ .Tanh = {} },
            .Softmax => return Layer{ .Softmax = {} },
        }
    }

    pub fn parameters(self: *Layer, allocator: Allocator) !std.ArrayListUnmanaged(*Tensor(f32)) {
        var list = std.ArrayListUnmanaged(*Tensor(f32)){};
        switch (self.*) {
            .Dense => |*l| {
                try list.append(allocator, l.weights);
                try list.append(allocator, l.bias);
            },
            else => {},
        }
        return list;
    }
};

/// Represents a flatten layer.
///
/// Flattens the input tensor into a 2D matrix (batch_size, features).
/// This is typically used to transition from convolutional layers to dense layers.
pub const Flatten = struct {
    /// Initializes a new Flatten layer.
    ///
    /// Returns:
    ///     A new Flatten layer instance.
    ///
    /// Example:
    /// ```zig
    /// var layer = Flatten.init();
    /// ```
    pub fn init() Flatten {
        return Flatten{};
    }

    /// Performs the forward pass of the layer.
    ///
    /// Reshapes the input array to (batch_size, -1).
    ///
    /// Arguments:
    ///     self: The Flatten layer instance.
    ///     allocator: The allocator to use for the output array.
    ///     input: The input array of shape (batch_size, ...).
    ///
    /// Returns:
    ///     The flattened output array of shape (batch_size, total_features).
    ///
    /// Example:
    /// ```zig
    /// var layer = layers.Flatten.init();
    /// var input = try NDArray(f32).init(allocator, &.{2, 3, 4}, &.{...}); // 2x3x4 input
    /// defer input.deinit();
    ///
    /// var output = try layer.forward(allocator, &input);
    /// defer output.deinit();
    /// // output shape is (2, 12)
    /// ```
    pub fn forward(self: *Flatten, allocator: Allocator, input: *const NDArray(f32)) !NDArray(f32) {
        _ = self;
        if (input.rank() < 2) return core.Error.RankMismatch;
        const batch_size = input.shape[0];
        const features = input.size() / batch_size;
        return input.reshape(allocator, &.{ batch_size, features });
    }
};

/// Represents a dropout layer for regularization during training.
///
/// Dropout randomly sets a fraction of input units to 0 at each update during training time,
/// which helps prevent overfitting.
///
/// Logic:
/// mask = bernoulli(1 - rate)
/// output = (input * mask) / (1 - rate)
pub const Dropout = struct {
    rate: f32,
    mask: ?NDArray(f32),
    training: bool,
    rng: random.Random,

    /// Initializes a new Dropout layer with the specified dropout rate.
    ///
    /// Arguments:
    ///     rate: The fraction of the input units to drop (between 0 and 1).
    ///     seed: Random seed for reproducibility.
    ///
    /// Returns:
    ///     A new Dropout layer instance.
    ///
    /// Example:
    /// ```zig
    /// var layer = Dropout.init(0.5, 42);
    /// defer layer.deinit();
    /// ```
    pub fn init(rate: f32, seed: u64) Dropout {
        return Dropout{ .rate = rate, .mask = null, .training = true, .rng = random.Random.init(seed) };
    }

    /// Deinitializes the layer, freeing associated memory.
    pub fn deinit(self: *Dropout) void {
        if (self.mask) |*m| m.deinit();
    }

    /// Performs the forward pass of the layer.
    ///
    /// Applies dropout to the input array if in training mode.
    ///
    /// Arguments:
    ///     self: The Dropout layer instance.
    ///     allocator: The allocator to use for the output array.
    ///     input: The input array.
    ///
    /// Returns:
    ///     The output of the layer (with dropout applied if training).
    ///
    /// Example:
    /// ```zig
    /// var output = try layer.forward(allocator, &input);
    /// defer output.deinit();
    /// ```
    pub fn forward(self: *Dropout, allocator: Allocator, input: *const NDArray(f32)) !NDArray(f32) {
        if (!self.training) {
            return input.copy(allocator);
        }

        // Generate mask
        const mask = try NDArray(f32).init(allocator, input.shape);
        const rand = self.rng.prng.random();

        for (mask.data) |*val| {
            val.* = if (rand.float(f32) > self.rate) 1.0 else 0.0;
        }

        // Scale by 1/(1-rate) to maintain expected value
        const scale = 1.0 / (1.0 - self.rate);

        var result = try NDArray(f32).init(allocator, input.shape);
        for (input.data, 0..) |val, i| {
            result.data[i] = val * mask.data[i] * scale;
        }

        if (self.mask) |*m| m.deinit();
        self.mask = mask;

        return result;
    }
};

test "ml layers dense forward" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    var layer = try Dense.init(allocator, 2, 1, .XavierUniform);
    defer layer.deinit(allocator);
    layer.weights.data.fill(0.01);
    layer.bias.data.fill(0.0);

    // Weights initialized to 0.01, bias 0
    // Input [1, 1] -> 1*0.01 + 1*0.01 = 0.02
    var input_data = try NDArray(f32).init(allocator, &.{ 1, 2 });
    input_data.fill(1.0);
    var input = try Tensor(f32).init(allocator, input_data, false);
    defer input.deinit(allocator);

    var out = try layer.forward(allocator, input);
    defer out.deinit(allocator);

    try std.testing.expectApproxEqAbs((try out.data.get(&.{ 0, 0 })), 0.02, 1e-4);
}
