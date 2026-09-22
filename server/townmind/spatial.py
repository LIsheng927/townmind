"""空间网格索引：把地图切成 cell_size × cell_size 的格子，每个实体按坐标落到对应格子里。

查"附近有谁"原来的做法（Agent._nearby()）是拿一个人的坐标，跟当前所有人的坐标逐个算一次
距离——NPC 数量是 n，一次查询是 O(n)，n 个 NPC 各自每次决策都要查一次，一轮下来就是 O(n²)。
NPC 数量小的时候完全无感，数量一多（几十、上百）就会先在这里卡住。

用网格之后，查"附近有谁"只需要看自己所在格子、加上周围一圈相邻格子（格子边长取跟"多近算
附近"这个半径一样大的话，一圈相邻格子最多 9 个），不用扫全部实体。

正确性上最容易出错的地方：如果只看"自己所在的格子"，会漏掉那些实际距离仍在半径以内、
但恰好落在隔壁格子的人（正好卡在格子边界附近）——所以必须查周围一圈格子，不能只查自己
这一格，下面 tests/test_spatial.py 里专门有一个用例在验证这一点。
"""
import math
from collections import defaultdict


class SpatialGrid:
    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.cell_size = cell_size
        self._cells: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._pos: dict[str, tuple[float, float]] = {}

    def _cell_of(self, pos: tuple[float, float]) -> tuple[int, int]:
        return (math.floor(pos[0] / self.cell_size), math.floor(pos[1] / self.cell_size))

    def update(self, entity_id: str, pos: tuple[float, float]) -> None:
        """新增或更新一个实体的位置；如果它之前在别的格子，先把它从那个格子里挪走。"""
        old = self._pos.get(entity_id)
        if old is not None:
            self._discard(entity_id, self._cell_of(old))
        self._pos[entity_id] = pos
        self._cells[self._cell_of(pos)].add(entity_id)

    def remove(self, entity_id: str) -> None:
        pos = self._pos.pop(entity_id, None)
        if pos is not None:
            self._discard(entity_id, self._cell_of(pos))

    def _discard(self, entity_id: str, cell: tuple[int, int]) -> None:
        cell_set = self._cells.get(cell)
        if cell_set is not None:
            cell_set.discard(entity_id)
            if not cell_set:
                del self._cells[cell]

    def nearby(self, entity_id: str, radius: float) -> list[tuple[str, float]]:
        """半径内的其他实体，连同到它们的距离，跟一开始"跟所有人算一遍距离"的写法结果完全一致，
        只是不用扫全部。"""
        me = self._pos.get(entity_id)
        if me is None:
            return []
        cx, cz = self._cell_of(me)
        span = math.ceil(radius / self.cell_size)
        result: list[tuple[str, float]] = []
        for dx in range(-span, span + 1):
            for dz in range(-span, span + 1):
                for other in self._cells.get((cx + dx, cz + dz), ()):
                    if other == entity_id:
                        continue
                    d = math.dist(me, self._pos[other])
                    if d <= radius:
                        result.append((other, d))
        return result

    def __len__(self) -> int:
        return len(self._pos)
