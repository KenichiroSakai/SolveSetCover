from ortools.sat.python import cp_model
import sys

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

def solve_set_cover_cpsat(cover_sets, m, n):
    """
    Solve the set cover problem using Google OR-Tools CP-SAT solver.
    """
    model = cp_model.CpModel()

    # Variables: one for each cover set, indicating whether it is selected (1) or not (0)
    num_sets = len(cover_sets)
    selected = [model.NewBoolVar(f'set_{i}') for i in range(num_sets)]

    # Constraint: Every element in the m x n matrix must be covered
    full_set = (1 << (m * n)) - 1  # All bits set for m x n matrix
    for bit in range(m * n):
        if (full_set & (1 << bit)) != 0:  # Check if the bit is part of the full set
            model.Add(sum(selected[i] for i in range(num_sets) if (cover_sets[i] & (1 << bit)) != 0) >= 1)

    # Objective: Minimize the number of selected sets
    model.Minimize(sum(selected))

    # Solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print the solver status
    print("Solver status:", solver.StatusName(status))

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = [i for i in range(num_sets) if solver.Value(selected[i]) == 1]
        return solution
    else:
        return None

def main():
    # Read m and n from command line
    if len(sys.argv) != 3:
        print("Usage: python solve_set_cover_cpsat.py <m> <n>")
        return
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    if m <= 0 or n <= 0:
        print("m and n must be positive integers.")
        return

    # Generate cover sets
    cover_sets = generate_cover_sets(m, n)

    # Solve the set cover problem
    solution = solve_set_cover_cpsat(cover_sets, m, n)

    # Output the result
    if solution:
        print(f"Minimum cover found with {len(solution)} sets:")
        for i in solution:
            print(bin(cover_sets[i])[2:].zfill(m * n))
    else:
        print("No cover found.")

if __name__ == "__main__":
    main()