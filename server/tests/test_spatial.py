"""验证 SpatialGrid 的查询结果，跟最原始的"逐个算距离"写法完全一致——这是这个优化能不能
放心用的关键：算法换了，结果不能变，尤其是格子边界附近容易漏判的情况。"""
import math
import random

from townmind.spatial import SpatialGrid


def _naive_nearby(positions: dict[str, tuple[float, float]], me: str, radius: float) -> set[str]:
    p = positions[me]
    return {o for o, pos in positions.items() if o != me and math.dist(p, pos) <= radius}


def test_empty_grid_returns_nothing():
    grid = SpatialGrid(cell_size=5.0)
    assert grid.nearby("alice", 5.0) == []


def test_unknown_entity_returns_nothing():
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (0.0, 0.0))
    assert grid.nearby("bob", 5.0) == []


def test_finds_entity_in_same_cell():
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (0.0, 0.0))
    grid.update("bob", (1.0, 1.0))
    result = dict(grid.nearby("alice", 5.0))
    assert "bob" in result
    assert result["bob"] == math.dist((0.0, 0.0), (1.0, 1.0))


def test_ignores_entity_outside_radius_even_in_same_cell():
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (0.0, 0.0))
    grid.update("bob", (4.9, 0.0))  # 同一个格子里，但距离超过半径
    assert grid.nearby("alice", 2.0) == []


def test_finds_entity_just_across_a_cell_boundary():
    """这是最容易踩坑的场景：两个人分别落在相邻的两个格子里，但实际距离明明在半径以内。
    如果查询只看"自己所在的格子"、不看周围一圈，这个用例就会漏判。"""
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (4.9, 0.0))  # 格子 (0, 0)
    grid.update("bob", (5.1, 0.0))  # 格子 (1, 0)——隔壁格子，但只隔了 0.2 米
    result = dict(grid.nearby("alice", 5.0))
    assert "bob" in result


def test_update_moves_entity_between_cells():
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (0.0, 0.0))
    grid.update("bob", (0.0, 0.0))
    assert "bob" in dict(grid.nearby("alice", 1.0))
    grid.update("bob", (100.0, 100.0))  # 挪到很远的地方
    assert "bob" not in dict(grid.nearby("alice", 1.0))


def test_remove_drops_entity_from_results():
    grid = SpatialGrid(cell_size=5.0)
    grid.update("alice", (0.0, 0.0))
    grid.update("bob", (1.0, 0.0))
    grid.remove("bob")
    assert grid.nearby("alice", 5.0) == []
    assert len(grid) == 1


def test_matches_naive_all_pairs_scan_on_random_layout():
    """随机撒一堆点，网格查询和"跟所有人逐个算距离"的结果要完全一样——这是最有说服力的
    正确性验证：不是挑几个手工设计的例子凑巧对，是在很多随机布局下都对得上。"""
    rng = random.Random(42)
    grid = SpatialGrid(cell_size=5.0)
    positions: dict[str, tuple[float, float]] = {}
    for i in range(80):
        pos = (rng.uniform(-30, 30), rng.uniform(-30, 30))
        eid = f"e{i}"
        positions[eid] = pos
        grid.update(eid, pos)

    for radius in (1.0, 5.0, 12.0):
        for eid in positions:
            expected = _naive_nearby(positions, eid, radius)
            got = {o for o, _ in grid.nearby(eid, radius)}
            assert got == expected, f"radius={radius} entity={eid} 结果对不上"
