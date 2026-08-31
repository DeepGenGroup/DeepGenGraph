from ortools.sat.python import cp_model


def solve_attention_tile(
    D: int,
    seqLen : int = 4096,
    shm_limit: int = 65536,
    align_bm: int = 16,
    align_bn: int = 16,
    bm_upper: int = 1024,
    bn_upper: int = 1024
):
    """
    求解:
        2 * (BM + BN) * D * 2 + BM * BN * 4 <= shm_limit

    等价于:
        D * (BM + BN) + BM * BN <= shm_limit / 4

    目标:
        尽可能增大 BM 和 BN，这里使用 maximize(BM + BN)

    BM / BN 分别要求是 align_bm / align_bn 的倍数。
    """

    model = cp_model.CpModel()

    # 用整数变量表示 BM/BN 的倍数
    bm_idx = model.NewIntVar(1, bm_upper // align_bm, "bm_idx")
    bn_idx = model.NewIntVar(1, bn_upper // align_bn, "bn_idx")

    BM = model.NewIntVar(align_bm, bm_upper, "BM")
    BN = model.NewIntVar(align_bn, bn_upper, "BN")
    
    bm_candidates = [
        x for x in range(align_bm, bm_upper + 1, align_bm)
        if seqLen % x == 0
    ]

    bn_candidates = [
        x for x in range(align_bn, bn_upper + 1, align_bn)
        if seqLen % x == 0
    ]

    model.AddAllowedAssignments([BM], [[x] for x in bm_candidates])
    model.AddAllowedAssignments([BN], [[x] for x in bn_candidates])
    model.Add(BM == bm_idx * align_bm)
    model.Add(BN == bn_idx * align_bn)

    # CP-SAT 不允许直接 BM * BN，需要显式乘积变量
    BM_BN = model.NewIntVar(
        0,
        bm_upper * bn_upper,
        "BM_BN"
    )
    model.AddMultiplicationEquality(BM_BN, [BM, BN])

    # shm:
    #
    # 2 * (BM + BN) * D * 2
    # + BM * BN * 4
    # <= shm_limit
    #
    # 为避免除法，直接保留原式
    shm_usage = model.NewIntVar(
        0,
        shm_limit,
        "shm_usage"
    )

    model.Add(
        shm_usage ==
        4 * D * (BM + BN)
        + 4 * BM_BN
    )

    model.Add(shm_usage <= shm_limit)

    # 目标：BM、BN 尽可能大
    min_tile = model.NewIntVar(0, min(bm_upper, bn_upper), "min_tile")

    model.AddMinEquality(min_tile, [BM, BN])

    BIG_WEIGHT = 10000

    model.Maximize(
        min_tile * BIG_WEIGHT
        + BM
        + BN
    )
    
    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        print("No feasible solution.")
        return None

    bm = solver.Value(BM)
    bn = solver.Value(BN)
    shm = solver.Value(shm_usage)

    print(f"D          = {D}")
    print(f"BM         = {bm}")
    print(f"BN         = {bn}")
    print(f"BM + BN    = {bm + bn}")
    print(f"BM * BN    = {bm * bn}")
    print(f"SHM usage  = {shm} bytes")
    print(f"SHM remain = {shm_limit - shm} bytes")

    return bm, bn


if __name__ == "__main__":
    bm,bn = solve_attention_tile(
        D=128,
        align_bm=2,
        align_bn=2,
    )
    seqLen = 4096
    bx = seqLen / bm
    forKUb = seqLen / bn
    print(f"{bx=} , {forKUb=}")
