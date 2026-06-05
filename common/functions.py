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
