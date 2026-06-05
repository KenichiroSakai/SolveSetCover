import sys

from common.functions import generate_cover_sets

def greedy_set_cover(cover_sets, m, n):
    """
    Solve the set cover problem using a greedy algorithm.
    """
    full_set = (1 << (m * n)) - 1  # All bits set for m x n matrix
    covered = 0
    selected_sets = []

    while covered != full_set:
        # Find the set that covers the most uncovered elements
        best_set = None
        best_coverage = 0
        for cover in cover_sets:
            coverage = bin(cover & ~covered).count('1')  # Count newly covered bits
            if coverage > best_coverage:
                best_coverage = coverage
                best_set = cover

        if best_set is None:
            # No progress can be made, return failure
            return None

        # Add the best set to the solution
        selected_sets.append(best_set)
        covered |= best_set

    return selected_sets

def main():
    # Read m and n from command line
    if len(sys.argv) != 3:
        print("Usage: python solve_set_cover_greedy.py <m> <n>")
        return
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    if m <= 0 or n <= 0:
        print("m and n must be positive integers.")
        return

    # Generate cover sets
    cover_sets = generate_cover_sets(m, n)

    # Solve the set cover problem
    solution = greedy_set_cover(cover_sets, m, n)

    # Output the result
    if solution:
        print(f"Minimum cover found with {len(solution)} sets:")
        for s in solution:
            print(bin(s)[2:].zfill(m * n))
    else:
        print("No cover found.")

if __name__ == "__main__":
    main()