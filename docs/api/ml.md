# Machine Learning API Reference

## Models

### `Sequential`

A sequential container for neural network layers.

#### `init(allocator: Allocator) Sequential`
Initializes a new empty Sequential model.

#### `deinit(self: *Sequential, allocator: Allocator) void`
Deinitializes the model and frees all associated memory.

#### `add(self: *Sequential, allocator: Allocator, layer: Layer) !void`
Adds a layer to the end of the model.

#### `forward(self: *Sequential, allocator: Allocator, input: *Tensor(f32)) !*Tensor(f32)`
Performs a forward pass through all layers in the model.

#### `fit(self: *Sequential, allocator: Allocator, x_train: *Tensor(f32), y_train: *Tensor(f32), optimizer: anytype, loss_fn: anytype, epochs: usize) !void`
Trains the model for a fixed number of epochs.
- **optimizer**: An optimizer instance (e.g., `SGD`) with an `update` method.
- **loss_fn**: A function that takes `(allocator, prediction, target)` and returns a loss Tensor.

#### `train(self: *Sequential) void`
Sets the model and all its layers to training mode.

#### `eval(self: *Sequential) void`
Sets the model and all its layers to evaluation mode.

#### `save(self: *Sequential, allocator: Allocator, path: []const u8) !void`
Saves the model architecture and weights to a binary file.

#### `load(allocator: Allocator, path: []const u8) !Sequential`
Loads a model from a binary file.

## Layers

### `Dense`

A fully connected layer.

#### `init(allocator: Allocator, input_dim: usize, output_dim: usize, init_method: InitMethod) !Dense`
Initializes a dense layer.
- **init_method**: `.RandomUniform`, `.XavierUniform`, or `.HeNormal`.

### `Layer` (Union)

A tagged union representing any supported layer type.

- `.Dense`: `Dense` struct.
- `.ReLU`: `void`.
- `.Sigmoid`: `void`.
- `.Tanh`: `void`.
- `.Softmax`: `void`.
- `.Dropout`: `Dropout` struct.

## Optimizers

### `SGD`

Stochastic Gradient Descent optimizer.

#### `init(lr: f32) SGD`
Initializes SGD with a learning rate.

#### `update(self: *SGD, param: *Tensor(f32)) void`
Updates a parameter tensor using its gradient.
