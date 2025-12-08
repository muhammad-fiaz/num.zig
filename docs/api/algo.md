# Algorithms API Reference

The `algo` module provides various data structures and algorithms.

## Search

### linearSearch

Perform a linear search for a target value.

```zig
pub fn linearSearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize
```

### binarySearch

Perform a binary search on a sorted array.

```zig
pub fn binarySearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize
```

### interpolationSearch

Perform an interpolation search on a sorted array.

```zig
pub fn interpolationSearch(allocator: Allocator, comptime T: type, a: *const NDArray(T), target: T) !?usize
```

## Data Structures

### Stack

A Last-In-First-Out (LIFO) stack.

```zig
pub fn Stack(comptime T: type) type
```

- `init()`: Initialize the stack.
- `deinit(allocator)`: Deinitialize the stack.
- `push(allocator, value)`: Push a value onto the stack.
- `pop()`: Pop a value from the stack.
- `peek()`: Peek at the top value.
- `isEmpty()`: Check if the stack is empty.

### Queue

A First-In-First-Out (FIFO) queue.

```zig
pub fn Queue(comptime T: type) type
```

- `init()`: Initialize the queue.
- `deinit(allocator)`: Deinitialize the queue.
- `enqueue(allocator, value)`: Add a value to the queue.
- `dequeue(allocator)`: Remove a value from the queue.
- `isEmpty()`: Check if the queue is empty.

### LinkedList

A singly linked list.

```zig
pub fn LinkedList(comptime T: type) type
```

- `init()`: Initialize the list.
- `deinit(allocator)`: Deinitialize the list.
- `append(allocator, value)`: Append a value.
- `prepend(allocator, value)`: Prepend a value.
- `find(value)`: Find a node.
- `delete(allocator, value)`: Delete a value.

### DoublyLinkedList

A doubly linked list.

```zig
pub fn DoublyLinkedList(comptime T: type) type
```

### CircularLinkedList

A circular linked list.

```zig
pub fn CircularLinkedList(comptime T: type) type
```

### PriorityQueue

A priority queue.

```zig
pub fn PriorityQueue(comptime T: type, comptime compare: fn (void, T, T) bool) type
```

### HashSet

A hash set.

```zig
pub fn HashSet(comptime T: type) type
```

## Graph

### Graph

A graph data structure.

```zig
pub fn Graph(comptime T: type) type
```

- `init()`: Initialize the graph.
- `deinit(allocator)`: Deinitialize the graph.
- `addVertex(allocator, v)`: Add a vertex.
- `addEdge(allocator, u, v)`: Add an edge.
- `bfs(allocator, start)`: Perform Breadth-First Search.
- `dfs(allocator, start)`: Perform Depth-First Search.

## Backtracking

### subsetSum

Solve the subset sum problem.

```zig
pub fn subsetSum(allocator: Allocator, set: *const NDArray(i32), target: i32) !bool
```
