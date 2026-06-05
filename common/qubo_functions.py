from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
import math, random

# =========================
# 1) QUBO 係数の生成
# =========================
def qubo_from_reach(
    reach: List[Set[int]],
    lam: float = 1.0,          # 親1台の一律コスト（重み付きにしないならこれ）
    A: float = 20.0,           # 罰則係数（lam の 10〜100倍くらいから調整）
    k: int = 1,                # k重被覆：各子を少なくとも k 回
    costs: Optional[List[float]] = None,   # 親ごとコスト c_i（指定時 lam 無視）
    include_self: bool = True, # reach[i] に i 自身をカバーとして含める
) -> Tuple[Dict[int,float], Dict[Tuple[int,int],float], float, int]:
    """
    QUBO:  minimize  const + sum_i b[i] x_i + sum_{i<l} Q[(i,l)] x_i x_l
    戻り値: (b, Q, const, num_parents)
    """
    N = len(reach)
    if costs is None:
        costs = [lam] * N
    else:
        assert len(costs) == N

    # reach を防御コピーしつつ自セルを含める
    R = [set(s) for s in reach]
    if include_self:
        for i in range(N):
            R[i].add(i)

    # 子集合 Universe を決定
    max_j = max((max(s) if s else -1) for s in R)
    if max_j < 0:
        raise ValueError("reach が空です。到達先が1つもありません。")
    M = max_j + 1  # 子の総数（0..max_j）

    # 子 j をカバー可能な親のリスト Cj を作る
    Cj: List[List[int]] = [[] for _ in range(M)]
    for i, s in enumerate(R):
        for j in s:
            if not (0 <= j < M):
                raise ValueError(f"子 idx {j} が範囲外です (0..{M-1})")
            Cj[j].append(i)
    for j in range(M):
        if not Cj[j]:
            raise ValueError(f"子 {j} をカバーできる親候補が存在しません。")

    # 係数
    b: Dict[int, float] = {i: float(costs[i]) for i in range(N)}  # 線形は親コストから開始
    Q: Dict[Tuple[int, int], float] = defaultdict(float)
    const = float(A * (k ** 2) * M)

    # j ごとに展開（k重被覆版の展開）
    # const += A * k^2
    # 線形: (-2kA) + A（x_i^2=x_i の寄与） -> 合算で (A - 2kA) = A*(1-2k)
    # 二次: 2A * x_i x_l
    for covs in Cj:
        for i in covs:
            b[i] += A * (1 - 2 * k)
        L = len(covs)
        for a in range(L):
            i = covs[a]
            for bidx in range(a + 1, L):
                l = covs[bidx]
                key = (i, l) if i < l else (l, i)
                Q[key] += 2.0 * A

    return b, dict(Q), const, N

# =========================
# 2) QUBO の簡易ソルバ（焼きなまし＋局所探索）
# =========================
def qubo_energy(x: List[int], b: Dict[int,float], Q: Dict[Tuple[int,int],float], const: float) -> float:
    # 事前に Q を i->[(l, coeff)] にしておくと速いが、簡潔さ優先
    val = const
    # 線形
    for i, xi in enumerate(x):
        if xi:
            val += b.get(i, 0.0)
    # 二次（i<l のみが入っている前提）
    for (i, l), w in Q.items():
        if x[i] and x[l]:
            val += w
    return val

def delta_energy_flip(i: int, x: List[int], b: Dict[int,float], Q: Dict[Tuple[int,int],float]) -> float:
    """
    x_i を 0->1 or 1->0 にフリップしたときのエネルギー差分（const は変わらない）
    ΔE = (1-2x_i) * [ b_i + sum_{l!=i} Q_{i,l} x_l ]
    Q は (min(i,l), max(i,l)) で格納されている前提
    """
    sign = 1 - 2 * x[i]  # x_i=0->+1, x_i=1->-1
    acc = b.get(i, 0.0)
    # Q の片側参照
    for (a, l), w in Q.items():
        if a == i and x[l]:
            acc += w
        elif l == i and x[a]:
            acc += w
    return sign * acc

def solve_qubo_sa(
    b: Dict[int,float],
    Q: Dict[Tuple[int,int],float],
    const: float,
    n: int,
    restarts: int = 20,
    sweeps: int = 2000,
    T0: float = 1.0,
    T_end: float = 1e-3,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    シンプルな焼きなまし＋貪欲局所探索。
    - restarts 回ランダム初期化
    - 各ランで徐々に温度を下げながら 単一ビットフリップ を受理
    - 最後に局所探索で改善
    """
    rng = random.Random(seed)
    best_x, best_E = None, float("inf")

    # 温度スケジュール（指数冷却）
    def temperature(t):
        if sweeps <= 1: return T_end
        r = t / (sweeps - 1)
        return T0 * (T_end / T0) ** r

    for _ in range(restarts):
        x = [rng.randint(0, 1) for _ in range(n)]
        # 温度付きスイープ
        for t in range(sweeps):
            T = temperature(t)
            i = rng.randrange(n)
            dE = delta_energy_flip(i, x, b, Q)
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-12)):
                x[i] ^= 1  # 受理

        # 局所探索（貪欲に ΔE<0 のフリップをし尽くす）
        improved = True
        while improved:
            improved = False
            # 無作為順で走査
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                dE = delta_energy_flip(i, x, b, Q)
                if dE < -1e-12:
                    x[i] ^= 1
                    improved = True

        E = qubo_energy(x, b, Q, const)
        if E < best_E:
            best_E, best_x = E, x[:]

    return best_x, best_E

# =========================
# 3) 被覆の検証と冗長除去
# =========================
def sets_to_bitmasks(reach: List[Set[int]], include_self: bool = True) -> List[int]:
    N = len(reach)
    S = [0] * N
    for i in range(N):
        mask = 0
        for j in reach[i]:
            mask |= (1 << j)
        if include_self:
            mask |= (1 << i)
        S[i] = mask
    return S

def verify_cover(S: List[int], chosen: Set[int]) -> bool:
    U = (1 << len(S)) - 1
    cov = 0
    for i in chosen:
        cov |= S[i]
    return cov == U

def prune_redundant(S: List[int], chosen: Set[int]) -> Set[int]:
    chosen = set(chosen)
    total = 0
    for i in chosen:
        total |= S[i]
    changed = True
    while changed:
        changed = False
        # 単純に一つ外しても被覆維持なら削る
        for i in list(chosen):
            without_i = 0
            for k in chosen:
                if k != i:
                    without_i |= S[k]
            if without_i == total:
                chosen.remove(i)
                total = without_i
                changed = True
    return chosen

