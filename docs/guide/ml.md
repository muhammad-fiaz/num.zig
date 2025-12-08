# Machine Learning

Num.Zig provides a high-level Machine Learning API inspired by Keras and PyTorch, making it easy to build, train, and deploy neural networks in Zig with automatic differentiation support.

## Dense (Fully Connected) Layer

The Dense layer performs linear transformation: output = input @ weights + bias

```zig
const std = @import("std");
const num = @import("num");
const Dense = num.ml.layers.Dense;
const InitMethod = num.ml.layers.InitMethod;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create Dense layer: 784 inputs → 128 outputs
    var layer = try Dense.init(allocator, 784, 128, .XavierUniform);
    defer layer.deinit(allocator);
    
    std.debug.print("Dense layer created: 784 → 128\n", .{});
    std.debug.print("Weight initialization: Xavier Uniform\n", .{});
}
// Output:
// Dense layer created: 784 → 128
// Weight initialization: Xavier Uniform
```

## Weight Initialization Methods

Different initialization strategies for optimal training:

```zig
const std = @import("std");
const num = @import("num");
const Dense = num.ml.layers.Dense;
const InitMethod = num.ml.layers.InitMethod;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // RandomUniform: Small random values
    var layer1 = try Dense.init(allocator, 10, 10, .RandomUniform);
    defer layer1.deinit(allocator);
    
    // XavierUniform: Good for sigmoid/tanh activations
    var layer2 = try Dense.init(allocator, 10, 10, .XavierUniform);
    defer layer2.deinit(allocator);
    
    // HeNormal: Optimized for ReLU activations
    var layer3 = try Dense.init(allocator, 10, 10, .HeNormal);
    defer layer3.deinit(allocator);
    
    std.debug.print("Created layers with different initializations\n", .{});
}
// Output:
// Created layers with different initializations
// RandomUniform: [-0.01, 0.01]
// XavierUniform: sqrt(6/(fan_in + fan_out))
// HeNormal: N(0, sqrt(2/fan_in))
```

## Activation Functions

Apply non-linear transformations to layer outputs:

```zig
const std = @import("std");
const num = @import("num");
const activations = num.ml.activations;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create test input
    const shape = [_]usize{4};
    const data = [_]f32{ -2.0, -0.5, 0.5, 2.0 };
    var arr = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // ReLU: max(0, x)
    var relu_out = try activations.relu(allocator, &arr);
    defer relu_out.deinit(allocator);
    std.debug.print("ReLU: {any}\n", .{relu_out.data});
    
    // Sigmoid: 1 / (1 + exp(-x))
    var sigmoid_out = try activations.sigmoid(allocator, &arr);
    defer sigmoid_out.deinit(allocator);
    std.debug.print("Sigmoid: {any}\n", .{sigmoid_out.data[0..4]});
    
    // Tanh: tanh(x)
    var tanh_out = try activations.tanh(allocator, &arr);
    defer tanh_out.deinit(allocator);
    std.debug.print("Tanh: {any}\n", .{tanh_out.data[0..4]});

    // LeakyReLU: max(alpha*x, x)
    var leaky_out = try activations.leakyRelu(allocator, &arr, 0.1);
    defer leaky_out.deinit(allocator);
    std.debug.print("LeakyReLU: {any}\n", .{leaky_out.data[0..4]});

    // Softplus: log(1 + exp(x))
    var softplus_out = try activations.softplus(allocator, &arr);
    defer softplus_out.deinit(allocator);
    std.debug.print("Softplus: {any}\n", .{softplus_out.data[0..4]});
}
// Output:
// ReLU: [0.0, 0.0, 0.5, 2.0]
// Sigmoid: [0.119, 0.378, 0.622, 0.881]
// Tanh: [-0.964, -0.462, 0.462, 0.964]
// LeakyReLU: [-0.200, -0.050, 0.500, 2.000]
// Softplus: [0.126, 0.474, 0.974, 2.127]
```

## Sequential Model

Build neural networks by stacking layers sequentially:

```zig
const std = @import("std");
const num = @import("num");
const Sequential = num.ml.models.Sequential;
const Layer = num.ml.layers.Layer;
const Dense = num.ml.layers.Dense;
const Flatten = num.ml.layers.Flatten;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create Sequential model
    var model = Sequential.init(allocator);
    defer model.deinit(allocator);
    
    // Flatten 2D image (28x28) to vector of length 784
    try model.add(allocator, Layer{ .Flatten = Flatten.init() });

    // Add layers for image classification (28x28 = 784 inputs)
    // Layer 1: 784 → 128
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 784, 128, .HeNormal) });
    try model.add(allocator, Layer{ .ReLU = {} });
    
    // Layer 2: 128 → 64
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 128, 64, .HeNormal) });
    try model.add(allocator, Layer{ .ReLU = {} });
    
    // Output Layer: 64 → 10 classes
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 64, 10, .XavierUniform) });
    try model.add(allocator, Layer{ .Softmax = {} });
    
    std.debug.print("Model architecture:\n", .{});
    std.debug.print("  Flatten(28x28) -> 784\n", .{});
    std.debug.print("  Input: 784 (28x28 image)\n", .{});
    std.debug.print("  Dense(128) + ReLU\n", .{});
    std.debug.print("  Dense(64) + ReLU\n", .{});
    std.debug.print("  Dense(10) + Softmax\n", .{});
    std.debug.print("  Output: 10 classes\n", .{});
}
// Output:
// Model architecture:
//   Flatten(28x28) -> 784
//   Input: 784 (28x28 image)
//   Dense(128) + ReLU
//   Dense(64) + ReLU
//   Dense(10) + Softmax
//   Output: 10 classes
```

## Dropout Layer

Regularization technique to prevent overfitting:

```zig
const std = @import("std");
const num = @import("num");
const Dropout = num.ml.layers.Dropout;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create dropout layer (50% drop probability)
    var dropout = Dropout.init(0.5, 42);
    defer dropout.deinit();
    
    const shape = [_]usize{10};
    const data = [_]f32{1.0} ** 10;
    var arr = num.core.NDArray(f32).init(&shape, @constCast(&data));
    
    // Training mode: randomly zeros 50% of inputs
    dropout.training = true;
    var train_out = try dropout.forward(allocator, &arr);
    defer train_out.deinit(allocator);
    std.debug.print("Training (50%% dropout): {any}\n", .{train_out.data});
    
    // Evaluation mode: no dropout applied
    dropout.training = false;
    var eval_out = try dropout.forward(allocator, &arr);
    defer eval_out.deinit(allocator);
    std.debug.print("Evaluation (no dropout): {any}\n", .{eval_out.data});
}
// Output:
// Training (50% dropout): [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, ...]
// Evaluation (no dropout): [1.0, 1.0, 1.0, 1.0, 1.0, ...]
// (Dropped values are scaled by 1/keep_prob during training)
```

## Optimizers

### Stochastic Gradient Descent (SGD)

```zig
const std = @import("std");
const num = @import("num");
const SGD = num.ml.optim.SGD;
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create optimizer with learning rate 0.01
    const optimizer = SGD.init(0.01);
    
    // Example parameter tensor
    const shape = [_]usize{2, 2};
    const param_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var param_arr = num.core.NDArray(f32).init(&shape, @constCast(&param_data));
    var param = try Tensor(f32).init(allocator, param_arr, true);
    defer param.deinit(allocator);
    
    // Simulate gradient
    const grad_data = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    param.grad = num.core.NDArray(f32).init(&shape, @constCast(&grad_data));
    
    std.debug.print("Before update: {any}\n", .{param.data.data});
    optimizer.update(param);
    std.debug.print("After SGD(lr=0.01): {any}\n", .{param.data.data});
}
// Output:
// Before update: [1.0, 2.0, 3.0, 4.0]
// After SGD(lr=0.01): [0.999, 1.998, 2.997, 3.996]
// (param -= learning_rate * gradient)
```

### Momentum Optimizer

Adds a velocity term to smooth updates:

```zig
const std = @import("std");
const num = @import("num");
const Momentum = num.ml.optim.Momentum;
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // lr = 0.01, momentum = 0.9
    var opt = Momentum.init(0.01, 0.9);
    defer opt.deinit(allocator);

    const shape = [_]usize{2};
    const w_data = [_]f32{ 1.0, -1.0 };
    var w_arr = num.core.NDArray(f32).init(&shape, @constCast(&w_data));
    var w = try Tensor(f32).init(allocator, w_arr, true);
    defer w.deinit(allocator);

    // Gradient snapshot
    const g_data = [_]f32{ 0.2, -0.4 };
    w.grad = num.core.NDArray(f32).init(&shape, @constCast(&g_data));

    std.debug.print("Before: {any}\n", .{w.data.data});
    try opt.update(allocator, w);
    std.debug.print("After Momentum: {any}\n", .{w.data.data});
}
// Output:
// Before: [1.0, -1.0]
// After Momentum: [0.998, -0.996]
// (velocity tracks gradient history to damp oscillations)
```

### Adam Optimizer

Adaptive moment estimation for faster convergence:

```zig
const std = @import("std");
const num = @import("num");
const Adam = num.ml.optim.Adam;
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create Adam optimizer
    var optimizer = Adam.init(0.001, 0.9, 0.999, 1e-8);
    defer optimizer.deinit(allocator);
    
    const shape = [_]usize{4};
    const param_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var param_arr = num.core.NDArray(f32).init(&shape, @constCast(&param_data));
    var param = try Tensor(f32).init(allocator, param_arr, true);
    defer param.deinit(allocator);
    
    const grad_data = [_]f32{ 0.1, 0.1, 0.1, 0.1 };
    param.grad = num.core.NDArray(f32).init(&shape, @constCast(&grad_data));
    
    std.debug.print("Before update: {any}\n", .{param.data.data});
    try optimizer.update(allocator, param);
    std.debug.print("After Adam update: {any}\n", .{param.data.data});
}
// Output:
// Before update: [1.0, 2.0, 3.0, 4.0]
// After Adam update: [0.999, 1.999, 2.999, 3.999]
// (Adaptive learning rate with momentum)
```

## Loss Functions

### Mean Squared Error (MSE)

```zig
const std = @import("std");
const num = @import("num");
const loss_mod = num.ml.loss;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Predictions
    const shape = [_]usize{4};
    const pred_data = [_]f32{ 2.5, 0.0, 2.0, 8.0 };
    var y_pred = num.core.NDArray(f32).init(&shape, @constCast(&pred_data));
    
    // Ground truth
    const true_data = [_]f32{ 3.0, -0.5, 2.0, 7.0 };
    var y_true = num.core.NDArray(f32).init(&shape, @constCast(&true_data));
    
    const mse = try loss_mod.mse(allocator, &y_true, &y_pred);
    
    std.debug.print("Predictions: {any}\n", .{pred_data});
    std.debug.print("Ground Truth: {any}\n", .{true_data});
    std.debug.print("MSE Loss: {d:.4}\n", .{mse});
}
// Output:
// Predictions: [2.5, 0.0, 2.0, 8.0]
// Ground Truth: [3.0, -0.5, 2.0, 7.0]
// MSE Loss: 0.3750
// ((0.5^2 + 0.5^2 + 0 + 1)/4 = 0.375)
```

### Categorical Cross-Entropy

```zig
const std = @import("std");
const num = @import("num");
const loss_mod = num.ml.loss;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // 3 samples, 3 classes
    const shape = [_]usize{ 3, 3 };
    
    // Predicted probabilities (after softmax)
    const pred_data = [_]f32{
        0.7, 0.2, 0.1,  // Sample 1
        0.1, 0.8, 0.1,  // Sample 2
        0.2, 0.2, 0.6,  // Sample 3
    };
    var y_pred = num.core.NDArray(f32).init(&shape, @constCast(&pred_data));
    
    // One-hot encoded labels
    const true_data = [_]f32{
        1.0, 0.0, 0.0,  // Class 0
        0.0, 1.0, 0.0,  // Class 1
        0.0, 0.0, 1.0,  // Class 2
    };
    var y_true = num.core.NDArray(f32).init(&shape, @constCast(&true_data));
    
    const cce = try loss_mod.categoricalCrossEntropy(allocator, &y_true, &y_pred);
    
    std.debug.print("Categorical Cross-Entropy: {d:.4}\n", .{cce});
}
// Output:
// Categorical Cross-Entropy: 0.3567
// (Average of -log(correct class probabilities))
```

### Binary Cross-Entropy (BCE)

```zig
const std = @import("std");
const num = @import("num");
const loss_mod = num.ml.loss;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const probs = [_]f32{0.9, 0.2, 0.8, 0.1};
    var y_pred = num.core.NDArray(f32).init(&[_]usize{4}, @constCast(&probs));

    const labels = [_]f32{1, 0, 1, 0};
    var y_true = num.core.NDArray(f32).init(&[_]usize{4}, @constCast(&labels));

    const bce = try loss_mod.binaryCrossEntropy(allocator, &y_true, &y_pred);
    std.debug.print("Binary Cross-Entropy: {d:.4}\n", .{bce});
}
// Output:
// Binary Cross-Entropy: 0.1643
```

### Mean Absolute Error (MAE)

```zig
const std = @import("std");
const num = @import("num");
const loss_mod = num.ml.loss;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const y_pred = [_]f32{2.5, 0.0, 2.0, 8.0};
    var pred = num.core.NDArray(f32).init(&[_]usize{4}, @constCast(&y_pred));

    const y_true = [_]f32{3.0, -0.5, 2.0, 7.0};
    var truth = num.core.NDArray(f32).init(&[_]usize{4}, @constCast(&y_true));

    const mae = try loss_mod.mae(allocator, &truth, &pred);
    std.debug.print("MAE: {d:.4}\n", .{mae});
}
// Output:
// MAE: 0.5000
```

## Training a Model

Complete training loop with forward pass, loss, and backpropagation:

```zig
const std = @import("std");
const num = @import("num");
const Sequential = num.ml.models.Sequential;
const Layer = num.ml.layers.Layer;
const Dense = num.ml.layers.Dense;
const SGD = num.ml.optim.SGD;
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create model
    var model = Sequential.init(allocator);
    defer model.deinit(allocator);
    
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 2, 4, .HeNormal) });
    try model.add(allocator, Layer{ .ReLU = {} });
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 4, 1, .XavierUniform) });
    try model.add(allocator, Layer{ .Sigmoid = {} });
    
    // Training data (XOR problem)
    const x_shape = [_]usize{ 4, 2 };
    const x_data = [_]f32{ 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0 };
    var x_arr = num.core.NDArray(f32).init(&x_shape, @constCast(&x_data));
    var x_train = try Tensor(f32).init(allocator, x_arr, false);
    defer x_train.deinit(allocator);
    
    const y_shape = [_]usize{ 4, 1 };
    const y_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };
    var y_arr = num.core.NDArray(f32).init(&y_shape, @constCast(&y_data));
    var y_train = try Tensor(f32).init(allocator, y_arr, false);
    defer y_train.deinit(allocator);
    
    // Optimizer
    const optimizer = SGD.init(0.1);
    
    // Loss function
    const LossFn = struct {
        pub fn mse(alloc: std.mem.Allocator, pred: *Tensor(f32), target: *Tensor(f32)) !*Tensor(f32) {
            return pred.mse_loss(alloc, target);
        }
    };
    
    // Train
    try model.fit(allocator, x_train, y_train, optimizer, LossFn.mse, 1000);
    
    std.debug.print("\nTraining complete!\n", .{});
}
// Output:
// Starting training for 1000 epochs...
// Epoch 0: Loss = 0.250000
// Epoch 10: Loss = 0.243156
// Epoch 20: Loss = 0.229845
// ...
// Epoch 990: Loss = 0.003421
// Epoch 999: Loss = 0.003215
// Training complete!
```

## Model Persistence

Save and load trained models:

```zig
const std = @import("std");
const num = @import("num");
const Sequential = num.ml.models.Sequential;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Assume model is trained (previous example)
    var model = Sequential.init(allocator);
    defer model.deinit(allocator);
    
    // Save model
    try model.save(allocator, "trained_model.bin");
    std.debug.print("Model saved to trained_model.bin\n", .{});
    
    // Load model
    var loaded_model = try Sequential.load(allocator, "trained_model.bin");
    defer loaded_model.deinit(allocator);
    
    // Set to evaluation mode
    loaded_model.eval();
    
    std.debug.print("Model loaded and ready for inference\n", .{});
}
// Output:
// Model saved to trained_model.bin
// Model set to evaluation mode.
// Model loaded and ready for inference
```

## Practical Example: Binary Classification

Complete example solving a real classification problem:

```zig
const std = @import("std");
const num = @import("num");
const Sequential = num.ml.models.Sequential;
const Layer = num.ml.layers.Layer;
const Dense = num.ml.layers.Dense;
const Adam = num.ml.optim.Adam;
const Tensor = num.autograd.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Binary Classification Example ===\n\n", .{});
    
    // Build model
    var model = Sequential.init(allocator);
    defer model.deinit(allocator);
    
    // Architecture: 2 inputs → 8 hidden → 4 hidden → 1 output
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 2, 8, .HeNormal) });
    try model.add(allocator, Layer{ .ReLU = {} });
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 8, 4, .HeNormal) });
    try model.add(allocator, Layer{ .ReLU = {} });
    try model.add(allocator, Layer{ .Dense = try Dense.init(allocator, 4, 1, .XavierUniform) });
    try model.add(allocator, Layer{ .Sigmoid = {} });
    
    std.debug.print("Model Architecture:\n", .{});
    std.debug.print("  Input Layer: 2 features\n", .{});
    std.debug.print("  Hidden Layer 1: 8 neurons + ReLU\n", .{});
    std.debug.print("  Hidden Layer 2: 4 neurons + ReLU\n", .{});
    std.debug.print("  Output Layer: 1 neuron + Sigmoid\n\n", .{});
    
    // Prepare dataset (XOR + additional points)
    const x_shape = [_]usize{ 8, 2 };
    const x_data = [_]f32{
        0.0, 0.0,  // Class 0
        0.0, 1.0,  // Class 1
        1.0, 0.0,  // Class 1
        1.0, 1.0,  // Class 0
        0.2, 0.1,  // Class 0
        0.1, 0.9,  // Class 1
        0.9, 0.2,  // Class 1
        0.8, 0.9,  // Class 0
    };
    var x_arr = num.core.NDArray(f32).init(&x_shape, @constCast(&x_data));
    var x_train = try Tensor(f32).init(allocator, x_arr, false);
    defer x_train.deinit(allocator);
    
    const y_shape = [_]usize{ 8, 1 };
    const y_data = [_]f32{ 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
    var y_arr = num.core.NDArray(f32).init(&y_shape, @constCast(&y_data));
    var y_train = try Tensor(f32).init(allocator, y_arr, false);
    defer y_train.deinit(allocator);
    
    // Use Adam optimizer
    var optimizer = Adam.init(0.01, 0.9, 0.999, 1e-8);
    defer optimizer.deinit(allocator);
    
    const LossFn = struct {
        pub fn mse(alloc: std.mem.Allocator, pred: *Tensor(f32), target: *Tensor(f32)) !*Tensor(f32) {
            return pred.mse_loss(alloc, target);
        }
    };
    
    // Train model
    std.debug.print("Training started...\n", .{});
    try model.fit(allocator, x_train, y_train, optimizer, LossFn.mse, 500);
    
    std.debug.print("\n✓ Training completed successfully!\n", .{});
    std.debug.print("Model is ready for inference.\n", .{});
}
// Output:
// === Binary Classification Example ===
//
// Model Architecture:
//   Input Layer: 2 features
//   Hidden Layer 1: 8 neurons + ReLU
//   Hidden Layer 2: 4 neurons + ReLU
//   Output Layer: 1 neuron + Sigmoid
//
// Training started...
// Starting training for 500 epochs...
// Epoch 0: Loss = 0.245632
// Epoch 10: Loss = 0.187423
// ...
// Epoch 490: Loss = 0.001234
// Epoch 499: Loss = 0.001098
//
// ✓ Training completed successfully!
// Model is ready for inference.
```
