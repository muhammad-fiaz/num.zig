# Machine Learning Guide

NumZig provides basic building blocks for machine learning, including layers, activations, loss functions, and optimizers.

## Layers

Layers are the core building blocks of neural networks.

### Dense Layer

A fully connected layer.

```zig
const num = @import("num");
const Dense = num.ml.layers.Dense;

// Initialize a dense layer with 10 input features and 5 output features
var layer = try Dense.init(allocator, 10, 5, .XavierUniform);
defer layer.deinit();

// Forward pass
var output = try layer.forward(allocator, &input);
defer output.deinit();
```

### Dropout Layer

A dropout layer for regularization.

```zig
const Dropout = num.ml.layers.Dropout;

// Initialize dropout with 0.5 rate
var dropout = Dropout.init(0.5);
defer dropout.deinit();

// Forward pass (training=true)
var output = try dropout.forward(allocator, &input, true);
defer output.deinit();
```

## Activations

Activation functions introduce non-linearity.

```zig
const activations = num.ml.activations;

// ReLU
var relu_out = try activations.relu(allocator, &input);

// Sigmoid
var sigmoid_out = try activations.sigmoid(allocator, &input);

// Softmax (usually for the last layer)
var probs = try activations.softmax(allocator, &logits, null); // null for last axis
```

## Loss Functions

Loss functions measure how well the model predicts the target.

```zig
const loss = num.ml.loss;

// Mean Squared Error (Regression)
const mse_val = try loss.mse(allocator, &y_true, &y_pred);

// Categorical Cross-Entropy (Classification)
const cce_val = try loss.categoricalCrossEntropy(allocator, &y_true, &y_pred);
```

## Optimizers

Optimizers update the model parameters to minimize the loss.

### SGD

```zig
const SGD = num.ml.optim.SGD;
var optimizer = SGD.init(0.01); // learning rate = 0.01

// Update parameters
optimizer.update(&weights, &grads);
```

### Adam

```zig
const Adam = num.ml.optim.Adam;
var optimizer = Adam.init(0.001, 0.9, 0.999, 1e-8);
defer optimizer.deinit();

// Update parameters
try optimizer.update(allocator, &weights, &grads);
```

## Example: Simple Training Loop

```zig
// ... setup data and model ...

var optimizer = num.ml.optim.SGD.init(0.01);

for (0..epochs) |epoch| {
    // Forward pass
    var hidden = try layer1.forward(allocator, &inputs);
    var output = try layer2.forward(allocator, &hidden);

    // Compute Loss
    const loss_val = try num.ml.loss.mse(allocator, &targets, &output);
    std.debug.print("Epoch {d}, Loss: {d}\n", .{epoch, loss_val});

    // Backward pass (simplified, assuming you have gradients)
    // ... compute gradients ...

    // Update weights
    optimizer.update(&layer1.weights, &grad_w1);
    optimizer.update(&layer2.weights, &grad_w2);
}
```
