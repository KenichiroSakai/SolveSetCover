import sys
from typing import List, Set
from common.qubo_functions import qubo_from_reach, solve_qubo_sa, sets_to_bitmasks, verify_cover, prune_redundant
from common.functions import generate_cover_sets

def bitmasks_to_sets(bitmasks: List[int], m: int, n: int) -> List[Set[int]]:
    """
    Convert a list of bitmasks to a list of sets of indices.

    Args:
        bitmasks (List[int]): List of bitmasks representing coverage.
        m (int): Number of rows in the grid.
        n (int): Number of columns in the grid.

    Returns:
        List[Set[int]]: List of sets where each set contains the indices covered by a parent.
    """
    sets = []
    for bitmask in bitmasks:
        covered = {i for i in range(m * n) if bitmask & (1 << i)}
        sets.append(covered)
    return sets

def solve_set_cover_qubo(reach: List[Set[int]], lam: float = 1.0, A: float = 30.0, k: int = 1):
    """
    Solve the set cover problem using QUBO formulation and simulated annealing.

    Args:
        reach (List[Set[int]]): List where each index represents a parent and the set contains the children it can cover.
        lam (float): Cost weight for each parent.
        A (float): Penalty coefficient for uncovered children.
        k (int): Minimum coverage requirement for each child.

    Returns:
        Tuple[Set[int], bool]: The set of chosen parents and whether the solution covers all children.
    """
    # Generate QUBO coefficients
    b, Q, const, N = qubo_from_reach(reach, lam=lam, A=A, k=k, include_self=True)

    # Solve QUBO using simulated annealing
    x, energy = solve_qubo_sa(b, Q, const, n=N, restarts=30, sweeps=3000, T0=1.5, T_end=1e-4, seed=42)

    # Extract chosen parents from solution
    chosen = {i for i, xi in enumerate(x) if xi == 1}

    # Verify coverage and prune redundant parents
    S = sets_to_bitmasks(reach, include_self=True)
    ok = verify_cover(S, chosen)
    chosen = prune_redundant(S, chosen)

    return chosen, ok

if __name__ == "__main__":
    # Read m and n from command line
    if len(sys.argv) != 3:
        print("Usage: python solve_set_cover_qubo.py <m> <n>")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])

    if m <= 0 or n <= 0:
        print("m and n must be positive integers.")
        sys.exit(1)

    # Generate reach using generate_cover_sets
    bitmasks = generate_cover_sets(m, n)
    reach = bitmasks_to_sets(bitmasks, m, n)

    # Solve set cover problem using QUBO
    chosen, ok = solve_set_cover_qubo(reach, lam=1.0, A=30.0, k=1)

    if ok:
        print(f"Minimum cover found with {len(chosen)} sets:")
        for i in sorted(chosen):
            print(bin(bitmasks[i])[2:].zfill(m * n))
    else:
        print("No cover found.")