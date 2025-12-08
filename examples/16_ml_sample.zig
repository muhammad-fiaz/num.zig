const std = @import("std");
const num = @import("num");
const Tensor = num.autograd.Tensor;
const NDArray = num.core.NDArray;
const Dense = num.ml.layers.Dense;
const Layer = num.ml.layers.Layer;
const Sequential = num.ml.models.Sequential;
const SGD = num.ml.optim.SGD;

const LossWrapper = struct {
    pub fn mse(alloc: std.mem.Allocator, pred: *Tensor(f32), target: *Tensor(f32)) !*Tensor(f32) {
        return pred.mse_loss(alloc, target);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Training XOR Neural Network using Sequential API...\n", .{});

    // 1. Prepare Data (XOR)
    // X: (4, 2)
    var x_data = try NDArray(f32).init(allocator, &.{ 4, 2 });
    x_data.data[0] = 0;
    x_data.data[1] = 0;
    x_data.data[2] = 0;
    x_data.data[3] = 1;
    x_data.data[4] = 1;
    x_data.data[5] = 0;
    x_data.data[6] = 1;
    x_data.data[7] = 1;
    const x = try Tensor(f32).init(allocator, x_data, false);
    defer x.deinit(allocator);

    // Y: (4, 1)
    var y_data = try NDArray(f32).init(allocator, &.{ 4, 1 });
    y_data.data[0] = 0;
    y_data.data[1] = 1;
    y_data.data[2] = 1;
    y_data.data[3] = 0;
    const y = try Tensor(f32).init(allocator, y_data, false);
    defer y.deinit(allocator);

    // 2. Define Model
    var model = Sequential.init(allocator);
    defer model.deinit(allocator);

    // Hidden Layer: 2 inputs -> 4 hidden units
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 2, 4, .XavierUniform) });
    try model.add(allocator, Layer{ .Tanh = {} });

    // Output Layer: 4 hidden units -> 1 output
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 4, 1, .XavierUniform) });
    try model.add(allocator, Layer{ .Sigmoid = {} });

    // 3. Define Optimizer
    const optimizer = SGD.init(0.1);

    // 4. Train
    try model.fit(allocator, x, y, optimizer, LossWrapper.mse, 2000);

    // 5. Save Model
    try model.save(allocator, "xor_model.bin");

    // 6. Load Model
    var loaded_model = try Sequential.load(allocator, "xor_model.bin");
    defer loaded_model.deinit(allocator);
    loaded_model.eval();

    // 7. Inference
    std.debug.print("\nInference Results (Loaded Model):\n", .{});

    // Create a temporary arena for inference to easily clean up intermediate tensors
    var infer_arena = std.heap.ArenaAllocator.init(allocator);
    defer infer_arena.deinit();
    const infer_alloc = infer_arena.allocator();

    const out = try loaded_model.forward(infer_alloc, x);

    for (0..4) |i| {
        const input1 = x.data.data[i * 2];
        const input2 = x.data.data[i * 2 + 1];
        const target = y.data.data[i];
        const prediction = out.data.data[i];
        std.debug.print("Input: ({d}, {d}) -> Target: {d} -> Pred: {d:.4}\n", .{ input1, input2, target, prediction });
    }
}
