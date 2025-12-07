# Algorithms Collection

The `num.algo` module provides a collection of standard algorithms implemented using `num.zig` data structures.

For convenience, common data structures are aliased directly under `num`:
- `num.Stack`
- `num.Queue`
- `num.LinkedList`
- `num.DoublyLinkedList`
- `num.CircularLinkedList`
- `num.Graph`

## Error Reporting

If you encounter any runtime or compile-time errors that seem like bugs, please report them at:
[https://github.com/muhammad-fiaz/num.zig/issues](https://github.com/muhammad-fiaz/num.zig/issues)

## Search

### Linear Search

Performs a linear search on an array.

**Logic:** Iterate and find index where value matches.

```zig
const idx = try num.algo.search.linearSearch(allocator, f32, &a, 5.0);
```

### Binary Search

Performs a binary search on a sorted 1D array.

**Logic:** Standard binary search.

```zig
const idx = try num.algo.search.binarySearch(allocator, f32, &a, 5.0);
```

## Data Structures

### Linked List

A generic singly linked list.

**Logic:** Standard linked list implementation.

```zig
var list = num.LinkedList(i32).init(allocator);
defer list.deinit();
try list.append(1);
```

### Doubly Linked List

A generic doubly linked list.

**Logic:** Nodes have next and prev pointers.

```zig
var dlist = num.DoublyLinkedList(i32).init(allocator);
```

### Circular Linked List

A generic circular linked list.

**Logic:** Tail points to head.

```zig
var clist = num.CircularLinkedList(i32).init(allocator);
```


### Stack

A generic LIFO stack.

**Logic:** Push/Pop from top.

```zig
var stack = num.Stack(i32).init(allocator);
try stack.push(1);
_ = stack.pop();
```

### Queue

A generic FIFO queue.

**Logic:** Enqueue at tail, Dequeue from head.

```zig
var queue = num.Queue(i32).init(allocator);
try queue.enqueue(1);
_ = queue.dequeue();
```

### Graph

A generic graph implementation (Adjacency List).

**Logic:** Vertices and Edges. Supports BFS and DFS.

```zig
var g = num.Graph(i32).init(allocator);
try g.addEdge(0, 1);
var path = try g.bfs(0);
```

## Backtracking

### Subset Sum

Solves the subset sum problem.

**Logic:** Backtracking.

```zig
const exists = try num.algo.backtracking.subsetSum(allocator, &set, 9);
```
