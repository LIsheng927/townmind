"""压测：模拟很多 NPC 同时在地图上，测"谁在附近"这个查询在 NPC 数量变大时，
朴素的"跟所有人算一遍距离"和空间网格两种做法，各自要花多少时间。

跑法（在 server 目录下，跟其它 evals 脚本一样用 -m 调用，这样 townmind 包才能被正确 import）：

    uv run python -m evals.scale_test
    uv run python -m evals.scale_test --counts 30 100 300 1000

为了让空间网格的优势能体现出来，NPC 是按"每人占的平均面积不变"撒在地图上的——地图大小
跟着 NPC 数量一起变大（模拟"这是个更大的开放世界，人口密度差不多"），而不是把所有人挤在
TownMind 现在那个 16x16 的小镇里。小镇本来就小，装不下太多人，硬塞进去的话谁跟谁都"在
附近"，体现不出网格的优势，也不真实——这是"如果小镇变成一个更大的开放世界"这个假设下的
压测，不是当前 TownMind 实际地图大小下的数字。

这个脚本不依赖 pydantic/fastapi 这些项目依赖，只用到 townmind.spatial（纯标准库），
所以哪怕项目依赖没装好，也能单独跑。
"""
import argparse
import math
import random
import time

from townmind.spatial import SpatialGrid

NEARBY_RADIUS = 5.0  # 要跟 townmind/agent.py 里的 NEARBY_RADIUS 保持一致
DENSITY = 25.0  # 平均每个 NPC 占的地图面积（约等于一个 cell_size=5 的格子大小）


def make_positions(n: int, rng: random.Random) -> dict[str, tuple[float, float]]:
    side = math.sqrt(n * DENSITY)
    return {f"npc{i}": (rng.uniform(-side / 2, side / 2), rng.uniform(-side / 2, side / 2)) for i in range(n)}


def naive_all(positions: dict[str, tuple[float, float]], radius: float) -> None:
    """朴素做法：每个 NPC 都跟所有人逐个算一遍距离——这是优化前 _nearby() 的写法。"""
    for me_id, me in positions.items():
        [(o, d) for o, pos in positions.items() if o != me_id and (d := math.dist(me, pos)) <= radius]


def grid_all(grid: SpatialGrid, ids: list[str], radius: float) -> None:
    for eid in ids:
        grid.nearby(eid, radius)


def run(n: int, rng: random.Random) -> tuple[float, float]:
    positions = make_positions(n, rng)

    t0 = time.perf_counter()
    naive_all(positions, NEARBY_RADIUS)
    naive_time = time.perf_counter() - t0

    grid = SpatialGrid(cell_size=NEARBY_RADIUS)
    for eid, pos in positions.items():
        grid.update(eid, pos)
    t0 = time.perf_counter()
    grid_all(grid, list(positions), NEARBY_RADIUS)
    grid_time = time.perf_counter() - t0

    return naive_time, grid_time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=int, nargs="+", default=[3, 30, 100, 300, 1000, 3000])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"{'NPC 数量':>10} {'朴素做法(全员一轮)':>20} {'空间网格(全员一轮)':>20} {'网格快多少倍':>12}")
    for n in args.counts:
        naive_time, grid_time = run(n, rng)
        speedup = naive_time / grid_time if grid_time > 0 else float("inf")
        print(f"{n:>10} {naive_time*1000:>17.2f}ms {grid_time*1000:>17.2f}ms {speedup:>10.1f}x")


if __name__ == "__main__":
    main()
