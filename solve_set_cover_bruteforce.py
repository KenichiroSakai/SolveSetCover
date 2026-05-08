import itertools
import sys
import time

def generate_cover_sets(m, n):
    """
    Generate the cover sets for an m x n matrix.
    Each cell covers itself and its adjacent cells (up, down, left, right).
    """
    cover_sets = []
    for i in range(m):
        for j in range(n):
            cover = 0
            # Current cell
            cover |= 1 << (i * n + j)
            # Up
            if i > 0:
                cover |= 1 << ((i - 1) * n + j)
            # Down
            if i < m - 1:
                cover |= 1 << ((i + 1) * n + j)
            # Left
            if j > 0:
                cover |= 1 << (i * n + (j - 1))
            # Right
            if j < n - 1:
                cover |= 1 << (i * n + (j + 1))
            cover_sets.append(cover)
    return cover_sets

def is_cover(cover_sets, m, n, timeout):
    """
    Check if a combination of cover sets covers the entire m x n matrix.
    """
    full_set = (1 << (m * n)) - 1  # All bits set for m x n matrix
    start_time = time.time()

    for r in range(1, len(cover_sets) + 1):
        for combination in itertools.combinations(cover_sets, r):
            # Check if the time limit has been exceeded
            if time.time() - start_time > timeout:
                print("Timeout reached. Unable to find a solution within the time limit.")
                return None

            combined_cover = 0
            for cover in combination:
                combined_cover |= cover
            if combined_cover == full_set:
                return combination
    return None

def main():
    # Read m and n from command line
    if len(sys.argv) != 3:
        print("Usage: python solve_set_cover_bruteforce.py <m> <n>")
        return
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    if m <= 0 or n <= 0:
        print("m and n must be positive integers.")
        return

    # Timeout in seconds
    TIMEOUT = 60

    # Generate cover sets
    cover_sets = generate_cover_sets(m, n)

    # Solve the set cover problem
    solution = is_cover(cover_sets, m, n, TIMEOUT)

    # Output the result
    if solution:
        print(f"Minimum cover found with {len(solution)} sets:")
        for s in solution:
            print(bin(s)[2:].zfill(m * n))
    else:
        print("No cover found.")

if __name__ == "__main__":
    main()