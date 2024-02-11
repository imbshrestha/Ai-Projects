import heapq

class Node:
    def __init__(self, state, parent=None, move=None, cost=0):
        self.state = state
        self.parent = parent
        self.move = move
        self.cost = cost
        self.heuristic = self.calculate_heuristic()

    def calculate_heuristic(self):
        # Define a heuristic function here
        # In Towers of Hanoi, a common heuristic is the number of disks not in their final position
        return sum(1 for i in range(len(self.state) - 1) if self.state[i] != i + 1)

    def __lt__(self, other):
        return self.cost + self.heuristic < other.cost + other.heuristic

def generate_successors(node):
    successors = []
    for i in range(len(node.state)):
        for j in range(len(node.state)):
            if i != j and (node.state[j] == 0 or node.state[i] < node.state[j]):
                new_state = list(node.state)
                new_state[j] = node.state[i]
                new_state[i] = 0
                successors.append(Node(tuple(new_state), parent=node, move=(i, j), cost=node.cost + 1))
    return successors

def solve(start_state, target_state):
    visited = set()
    heap = []
    heapq.heappush(heap, Node(start_state))

    while heap:
        node = heapq.heappop(heap)
        if node.state == target_state:
            return node

        visited.add(node.state)
        successors = generate_successors(node)
        for successor in successors:
            if successor.state not in visited:
                heapq.heappush(heap, successor)

    return None

def print_solution(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    for state in path:
        print(state)

def get_input():
    start_state = tuple(map(int, input("Enter the initial state of the Towers of Hanoi puzzle (comma-separated integers): ").split(',')))
    target_state = tuple(map(int, input("Enter the target state of the Towers of Hanoi puzzle (comma-separated integers): ").split(',')))
    return start_state, target_state

# Example usage
start_state, target_state = get_input()
solution_node = solve(start_state, target_state)
if solution_node:
    print("Solution found:")
    print_solution(solution_node)
else:
    print("No solution found.")
