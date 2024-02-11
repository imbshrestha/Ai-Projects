from simpleai.search import astar, SearchProblem
from typing import List, Tuple

class TowersOfHanoi(SearchProblem):
    def __init__(self, initial: Tuple[int, ...], goal: Tuple[int, ...]):
        """
        Initialize the Towers of Hanoi problem.

        Args:
        - initial: Initial state of the Towers of Hanoi puzzle
        - goal: Target state of the Towers of Hanoi puzzle
        """
        self.initial = initial
        self.goal = goal
        super().__init__(initial_state=initial)

    def actions(self, state: Tuple[int, ...]) -> List[Tuple[int, int]]:
        """
        Return a list of valid moves from the given state.

        Args:
        - state: Current state of the puzzle

        Returns:
        - List of valid moves (each move represented as a tuple of two peg indices)
        """
        actions = []
        for i in range(len(state)):
            for j in range(len(state)):
                if i != j and (state[j] == 0 or state[i] < state[j]):
                    actions.append((i, j))
        return actions

    def result(self, state: Tuple[int, ...], action: Tuple[int, int]) -> Tuple[int, ...]:
        """
        Apply the given action to the state and return the resulting state.

        Args:
        - state: Current state of the puzzle
        - action: Move to be applied (tuple of two peg indices)

        Returns:
        - Resulting state after applying the action
        """
        source, target = action
        new_state = list(state)
        new_state[target] = state[source]
        new_state[source] = 0
        return tuple(new_state)

    def is_goal(self, state: Tuple[int, ...]) -> bool:
        """
        Check if the given state is the goal state.

        Args:
        - state: Current state of the puzzle

        Returns:
        - True if the state is the goal state, False otherwise
        """
        return state == self.goal

    def heuristic(self, state: Tuple[int, ...]) -> int:
        """
        Calculate the heuristic value for the given state.

        Args:
        - state: Current state of the puzzle

        Returns:
        - Heuristic value (number of disks not in their final position)
        """
        return sum(1 for i in range(len(state) - 1) if state[i] != i + 1)

def get_input():
    start_state = tuple(map(int, input("Enter the initial state of the Towers of Hanoi puzzle (comma-separated integers): ").split(',')))
    target_state = tuple(map(int, input("Enter the target state of the Towers of Hanoi puzzle (comma-separated integers): ").split(',')))
    return start_state, target_state

# Example usage
start_state, target_state = get_input()
problem = TowersOfHanoi(initial=start_state, goal=target_state)
result = astar(problem)

# Print solution
if result:
    print("Solution found:")
    for action, state in result.path():
        print(state)
else:
    print("No solution found.")
