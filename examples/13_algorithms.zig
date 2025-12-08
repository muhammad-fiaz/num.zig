const std = @import("std");
const num = @import("num");
const print = std.debug.print;

pub fn main() !void {
    mainImpl() catch |err| {
        print("Runtime Error: {}\n", .{err});
        print("If you think this is a bug, please report it at: {s}\n", .{num.report_issue_url});
        return err;
    };
}

fn mainImpl() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Example 13: Algorithms Collection\n", .{});

    // 1. Linear Search
    var a = try num.core.NDArray(f32).init(allocator, &.{5});
    @memcpy(a.data, &[_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 });
    defer a.deinit(allocator);

    if (try num.algo.search.linearSearch(allocator, f32, &a, 30.0)) |idx| {
        print("Linear Search: Found 30.0 at index {}\n", .{idx});
    } else {
        print("Linear Search: 30.0 not found\n", .{});
    }

    // 2. Binary Search
    if (try num.algo.search.binarySearch(allocator, f32, &a, 40.0)) |idx| {
        print("Binary Search: Found 40.0 at index {}\n", .{idx});
    } else {
        print("Binary Search: 40.0 not found\n", .{});
    }

    // 3. Linked List
    var list = num.LinkedList(i32).init();
    defer list.deinit(allocator);

    try list.append(allocator, 1);
    try list.append(allocator, 2);
    try list.append(allocator, 3);
    print("Linked List: Appended 1, 2, 3\n", .{});

    if (list.find(2)) |_| {
        print("Linked List: Found 2\n", .{});
    }

    if (list.delete(allocator, 2)) {
        print("Linked List: Deleted 2\n", .{});
    }

    // 4. Backtracking (Subset Sum)
    var set = try num.core.NDArray(i32).init(allocator, &.{6});
    @memcpy(set.data, &[_]i32{ 3, 34, 4, 12, 5, 2 });
    defer set.deinit(allocator);
    const target = 9;

    const exists = try num.algo.backtracking.subsetSum(allocator, &set, target);
    print("Subset Sum: Subset summing to {} exists? {}\n", .{ target, exists });

    // 5. Doubly Linked List
    var dlist = num.DoublyLinkedList(i32).init(allocator);
    defer dlist.deinit();
    try dlist.append(10);
    try dlist.prepend(5);
    print("Doubly Linked List: Prepend 5, Append 10\n", .{});

    // 6. Stack
    var stack = num.Stack(i32).init();
    defer stack.deinit(allocator);
    try stack.push(allocator, 1);
    try stack.push(allocator, 2);
    print("Stack: Pop {}\n", .{stack.pop().?}); // 2

    // 7. Graph BFS
    var graph = num.Graph(i32).init();
    defer graph.deinit(allocator);
    try graph.addEdge(allocator, 0, 1);
    try graph.addEdge(allocator, 0, 2);
    try graph.addEdge(allocator, 1, 2);
    try graph.addEdge(allocator, 2, 0);
    try graph.addEdge(allocator, 2, 3);
    try graph.addEdge(allocator, 3, 3);

    var bfs_res = try graph.bfs(allocator, 2);
    defer bfs_res.deinit(allocator);
    print("Graph BFS from 2: ", .{});
    for (bfs_res.items) |v| print("{} ", .{v});
    print("\n", .{});
}
