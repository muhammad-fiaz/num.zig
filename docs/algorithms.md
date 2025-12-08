# Algorithms

Num.Zig provides a comprehensive collection of classic algorithms and data structures for computational problems.

## Search Algorithms

### Linear Search

Sequential search through an array - **O(n)** time complexity.

```zig
const std = @import("std");
const num = @import("num");
const search = num.algo.search;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	const arr = [_]i32{ 10, 23, 45, 70, 11, 15 };
	const target: i32 = 70;
    
	const result = try search.linearSearch(allocator, i32, &arr, target);
    
	if (result) |index| {
		std.debug.print("Found {d} at index {d}\n", .{ target, index });
	} else {
		std.debug.print("{d} not found\n", .{target});
	}
}
// Output:
// Found 70 at index 3
// Time Complexity: O(n) - checks each element sequentially
```

### Binary Search

Efficient search on sorted arrays - **O(log n)** time complexity.

```zig
const std = @import("std");
const num = @import("num");
const search = num.algo.search;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	// Array MUST be sorted
	const arr = [_]i32{ 2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78 };
	const target: i32 = 23;
    
	const result = try search.binarySearch(allocator, i32, &arr, target);
    
	if (result) |index| {
		std.debug.print("Found {d} at index {d}\n", .{ target, index });
	} else {
		std.debug.print("{d} not found\n", .{target});
	}
}
// Output:
// Found 23 at index 5
// Time Complexity: O(log n) - divides search space in half each iteration
// Space Complexity: O(1) - constant space
```

### Interpolation Search

Optimized for uniformly distributed sorted arrays - **O(log log n)** average case.

```zig
const std = @import("std");
const num = @import("num");
const search = num.algo.search;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	// Works best with uniformly distributed data
	const arr = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	const target: i32 = 70;
    
	const result = try search.interpolationSearch(allocator, i32, &arr, target);
    
	if (result) |index| {
		std.debug.print("Found {d} at index {d}\n", .{ target, index });
	}
}
// Output:
// Found 70 at index 6
// Best for uniformly distributed data
// Average: O(log log n), Worst: O(n)
```

## Sorting Algorithms

### Sort by Algorithm

Generic sorting with multiple algorithm choices:

```zig
const std = @import("std");
const num = @import("num");
const sort = num.algo.sort;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var arr = [_]i32{ 64, 34, 25, 12, 22, 11, 90 };
    
	std.debug.print("Original: {any}\n", .{arr});
    
	// Sort in-place
	try sort.sortByAlgo(i32, &arr, .QuickSort);
    
	std.debug.print("Sorted: {any}\n", .{arr});
}
// Output:
// Original: [64, 34, 25, 12, 22, 11, 90]
// Sorted: [11, 12, 22, 25, 34, 64, 90]
// QuickSort: Average O(n log n), Worst O(n²)
```

### Argsort

Returns indices that would sort the array:

```zig
const std = @import("std");
const num = @import("num");
const sort = num.algo.sort;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	const arr = [_]f32{ 3.5, 1.2, 4.8, 2.1 };
    
	const indices = try sort.argsort(allocator, f32, &arr);
	defer allocator.free(indices);
    
	std.debug.print("Array: {any}\n", .{arr});
	std.debug.print("Sorted indices: {any}\n", .{indices});
	std.debug.print("Values in sorted order: ", .{});
	for (indices) |idx| {
		std.debug.print("{d:.1} ", .{arr[idx]});
	}
	std.debug.print("\n", .{});
}
// Output:
// Array: [3.5, 1.2, 4.8, 2.1]
// Sorted indices: [1, 3, 0, 2]
// Values in sorted order: 1.2 2.1 3.5 4.8
```

### Nonzero Elements

Find indices of non-zero elements:

```zig
const std = @import("std");
const num = @import("num");
const sort = num.algo.sort;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	const arr = [_]i32{ 0, 5, 0, 0, 3, 0, 8 };
    
	const nz = try sort.nonzero(allocator, i32, &arr);
	defer allocator.free(nz);
    
	std.debug.print("Array: {any}\n", .{arr});
	std.debug.print("Non-zero indices: {any}\n", .{nz});
	std.debug.print("Non-zero values: ", .{});
	for (nz) |idx| {
		std.debug.print("{d} ", .{arr[idx]});
	}
	std.debug.print("\n", .{});
}
// Output:
// Array: [0, 5, 0, 0, 3, 0, 8]
// Non-zero indices: [1, 4, 6]
// Non-zero values: 5 3 8
```

## Graph Algorithms

Generic graph implementation with traversal algorithms:

### Breadth-First Search (BFS)

```zig
const std = @import("std");
const num = @import("num");
const Graph = num.algo.graph.Graph;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	// Create graph with integer vertices
	var graph = Graph(u32).init(allocator);
	defer graph.deinit();
    
	// Build graph:
	//    1 --- 2
	//    |     |
	//    3 --- 4 --- 5
	try graph.addVertex(1);
	try graph.addVertex(2);
	try graph.addVertex(3);
	try graph.addVertex(4);
	try graph.addVertex(5);
    
	try graph.addEdge(1, 2);
	try graph.addEdge(1, 3);
	try graph.addEdge(2, 4);
	try graph.addEdge(3, 4);
	try graph.addEdge(4, 5);
    
	std.debug.print("BFS from vertex 1:\n", .{});
	const bfs_result = try graph.bfs(allocator, 1);
	defer allocator.free(bfs_result);
    
	std.debug.print("Traversal order: ", .{});
	for (bfs_result) |vertex| {
		std.debug.print("{d} ", .{vertex});
	}
	std.debug.print("\n", .{});
}
// Output:
// BFS from vertex 1:
// Traversal order: 1 2 3 4 5
// (Level-by-level traversal: visits all neighbors before going deeper)
```

### Depth-First Search (DFS)

```zig
const std = @import("std");
const num = @import("num");
const Graph = num.algo.graph.Graph;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var graph = Graph(u32).init(allocator);
	defer graph.deinit();
    
	// Same graph structure as BFS example
	try graph.addVertex(1);
	try graph.addVertex(2);
	try graph.addVertex(3);
	try graph.addVertex(4);
	try graph.addVertex(5);
    
	try graph.addEdge(1, 2);
	try graph.addEdge(1, 3);
	try graph.addEdge(2, 4);
	try graph.addEdge(3, 4);
	try graph.addEdge(4, 5);
    
	std.debug.print("DFS from vertex 1:\n", .{});
	const dfs_result = try graph.dfs(allocator, 1);
	defer allocator.free(dfs_result);
    
	std.debug.print("Traversal order: ", .{});
	for (dfs_result) |vertex| {
		std.debug.print("{d} ", .{vertex});
	}
	std.debug.print("\n", .{});
}
// Output:
// DFS from vertex 1:
// Traversal order: 1 2 4 3 5
// (Goes as deep as possible before backtracking)
```

## Data Structures

### Stack (LIFO)

Last-In-First-Out data structure:

```zig
const std = @import("std");
const num = @import("num");
const Stack = num.algo.stack.Stack;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var stack = Stack(i32).init(allocator);
	defer stack.deinit();
    
	// Push elements
	try stack.push(10);
	try stack.push(20);
	try stack.push(30);
    
	std.debug.print("Stack size: {d}\n", .{stack.size()});
    
	// Pop elements (LIFO order)
	std.debug.print("Popped: {d}\n", .{try stack.pop()});
	std.debug.print("Popped: {d}\n", .{try stack.pop()});
    
	// Peek at top
	std.debug.print("Top element: {d}\n", .{try stack.peek()});
}
// Output:
// Stack size: 3
// Popped: 30
// Popped: 20
// Top element: 10
// (Last In, First Out - like a stack of plates)
```

### Queue (FIFO)

First-In-First-Out data structure:

```zig
const std = @import("std");
const num = @import("num");
const Queue = num.algo.queue.Queue;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var queue = Queue(i32).init(allocator);
	defer queue.deinit();
    
	// Enqueue elements
	try queue.enqueue(10);
	try queue.enqueue(20);
	try queue.enqueue(30);
    
	std.debug.print("Queue size: {d}\n", .{queue.size()});
	std.debug.print("Is empty: {}\n", .{queue.isEmpty()});
    
	// Dequeue elements (FIFO order)
	std.debug.print("Dequeued: {d}\n", .{try queue.dequeue()});
	std.debug.print("Dequeued: {d}\n", .{try queue.dequeue()});
}
// Output:
// Queue size: 3
// Is empty: false
// Dequeued: 10
// Dequeued: 20
// (First In, First Out - like a line at a store)
```

### Linked List

Singly linked list implementation:

```zig
const std = @import("std");
const num = @import("num");
const LinkedList = num.algo.list.LinkedList;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var list = LinkedList(i32).init(allocator);
	defer list.deinit();
    
	// Add elements
	try list.append(10);
	try list.append(20);
	try list.prepend(5);
	try list.insert(1, 15);  // Insert at index 1
    
	std.debug.print("List: ", .{});
	list.print();
	std.debug.print("\n", .{});
    
	// Remove element
	_ = try list.remove(2);
    
	std.debug.print("After removing index 2: ", .{});
	list.print();
	std.debug.print("\n", .{});
}
// Output:
// List: 5 -> 15 -> 10 -> 20 -> null
// After removing index 2: 5 -> 15 -> 20 -> null
```

### Doubly Linked List

Bidirectional linked list:

```zig
const std = @import("std");
const num = @import("num");
const DoublyLinkedList = num.algo.list.DoublyLinkedList;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var list = DoublyLinkedList(i32).init(allocator);
	defer list.deinit();
    
	try list.append(10);
	try list.append(20);
	try list.append(30);
    
	std.debug.print("Forward traversal: ", .{});
	list.print();
	std.debug.print("\n", .{});
}
// Output:
// Forward traversal: 10 <-> 20 <-> 30 <-> null
// (Each node has both next and prev pointers)
```

### Circular Linked List

List where last node points back to first:

```zig
const std = @import("std");
const num = @import("num");
const CircularLinkedList = num.algo.list.CircularLinkedList;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var list = CircularLinkedList(i32).init(allocator);
	defer list.deinit();
    
	try list.append(10);
	try list.append(20);
	try list.append(30);
    
	std.debug.print("Circular list: ", .{});
	list.print();
	std.debug.print("\n", .{});
}
// Output:
// Circular list: 10 -> 20 -> 30 -> (back to 10)
// (Last node connects to first, forming a circle)
```

## Collections

### Priority Queue

Heap-based priority queue (min-heap by default):

```zig
const std = @import("std");
const num = @import("num");
const PriorityQueue = num.algo.collections.PriorityQueue;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var pq = try PriorityQueue(i32).init(allocator);
	defer pq.deinit();
    
	// Insert elements
	try pq.add(30);
	try pq.add(10);
	try pq.add(50);
	try pq.add(20);
    
	std.debug.print("Priority Queue (min-heap):\n", .{});
	std.debug.print("Size: {d}\n", .{pq.size()});
    
	// Extract in priority order
	std.debug.print("Extracted: {d}\n", .{try pq.poll()});
	std.debug.print("Extracted: {d}\n", .{try pq.poll()});
	std.debug.print("Peek: {d}\n", .{try pq.peek()});
}
// Output:
// Priority Queue (min-heap):
// Size: 4
// Extracted: 10
// Extracted: 20
// Peek: 30
// (Always returns smallest element first)
```

### Hash Set

Unordered collection of unique elements:

```zig
const std = @import("std");
const num = @import("num");
const HashSet = num.algo.collections.HashSet;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	var set = HashSet(i32).init(allocator);
	defer set.deinit();
    
	// Add elements
	try set.add(10);
	try set.add(20);
	try set.add(10);  // Duplicate ignored
	try set.add(30);
    
	std.debug.print("Set size: {d}\n", .{set.size()});
	std.debug.print("Contains 20: {}\n", .{set.contains(20)});
	std.debug.print("Contains 40: {}\n", .{set.contains(40)});
    
	// Remove element
	try set.remove(20);
	std.debug.print("After removing 20, contains: {}\n", .{set.contains(20)});
}
// Output:
// Set size: 3
// Contains 20: true
// Contains 40: false
// After removing 20, contains: false
// (No duplicates, fast O(1) lookup)
```

## Backtracking Algorithms

### Subset Sum Problem

Find all subsets that sum to a target value:

```zig
const std = @import("std");
const num = @import("num");
const backtracking = num.algo.backtracking;

pub fn main() !void {
	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	defer _ = gpa.deinit();
	const allocator = gpa.allocator();

	const arr = [_]i32{ 3, 34, 4, 12, 5, 2 };
	const target: i32 = 9;
    
	std.debug.print("Finding subsets that sum to {d}:\n", .{target});
	std.debug.print("Array: {any}\n", .{arr});
    
	const results = try backtracking.subsetSum(allocator, &arr, target);
	defer {
		for (results) |subset| {
			allocator.free(subset);
		}
		allocator.free(results);
	}
    
	std.debug.print("\nSolutions found: {d}\n", .{results.len});
	for (results, 0..) |subset, i| {
		std.debug.print("Solution {d}: {any}\n", .{ i + 1, subset });
	}
}
// Output:
// Finding subsets that sum to 9:
// Array: [3, 34, 4, 12, 5, 2]
//
// Solutions found: 3
// Solution 1: [3, 4, 2]
// Solution 2: [4, 5]
// Solution 3: [3, 5, 2] (order may vary)
// (Explores all possible combinations using backtracking)
```

## Algorithm Complexity Summary

| Algorithm | Best Case | Average Case | Worst Case | Space |
|-----------|-----------|--------------|------------|-------|
| **Search** |
| Linear Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Interpolation Search | O(1) | O(log log n) | O(n) | O(1) |
| **Sort** |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| **Graph** |
| BFS | O(V+E) | O(V+E) | O(V+E) | O(V) |
| DFS | O(V+E) | O(V+E) | O(V+E) | O(V) |
| **Data Structures** |
| Stack (push/pop) | O(1) | O(1) | O(1) | O(n) |
| Queue (enqueue/dequeue) | O(1) | O(1) | O(1) | O(n) |
| Linked List (insert) | O(1) | O(1) | O(1) | O(n) |
| Priority Queue (add) | O(log n) | O(log n) | O(log n) | O(n) |
| Hash Set (add/contains) | O(1) | O(1) | O(n) | O(n) |

*V = vertices, E = edges*
