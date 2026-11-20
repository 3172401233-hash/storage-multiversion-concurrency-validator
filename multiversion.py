"""多版本对象存储并发校验的本地事件模型。

本模块不连接任何真实对象存储,只用本地事件序列模拟:
对象版本、ETag、写入序号、删除标记、读取条件(快照)、
副本位置与清理水位,从而计算每次读取的可见版本以及
并发操作后的最终目录状态。

语义约定:
- 当前读(无快照)只能看到写入序号最大的非删除标记版本;
  删除标记与普通旧版本对当前读不可见。
- 条件写必须绑定指定版本:仅当协调副本的当前版本恰好是
  所声明的基版本时写入才生效,否则抛出 ConditionalWriteFailed。
- 重复删除是幂等的:当前版本已是删除标记时不再产生新版本。
- 后台清理只能移除写入序号 <= 水位、且不是当前版本、
  且未被任何活跃快照引用的版本。
- 每个副本独立应用事件,因此同一时刻不同副本的可见状态
  可能不同(多副本差异)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


class ConditionalWriteFailed(Exception):
    """条件写声明的基版本与协调副本当前版本不一致。"""


@dataclass
class Version:
    """一个对象版本(或删除标记)。"""

    version_id: str
    etag: str
    seq: int
    delete_marker: bool = False


@dataclass
class Snapshot:
    """读快照:固定某个写入序号水位,读取据此可见的版本。"""

    snapshot_id: str
    seq: int


@dataclass
class Replica:
    """副本:独立持有已应用事件的版本集合。"""

    name: str
    versions: Dict[str, Version] = field(default_factory=dict)
    watermark: int = 0

    def current(self) -> Optional[Version]:
        """写入序号最大的版本(可能是删除标记)。"""
        if not self.versions:
            return None
        return max(self.versions.values(), key=lambda v: v.seq)

    def visible(self, as_of_seq: Optional[int] = None) -> Optional[Version]:
        """计算可见版本。

        as_of_seq 为 None 表示当前读;否则表示快照读。
        删除标记使其之前的版本不可见(返回 None);
        普通旧版本对超出其序号的读也不可见。
        """
        candidates = [
            v
            for v in self.versions.values()
            if as_of_seq is None or v.seq <= as_of_seq
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda v: v.seq)
        if latest.delete_marker:
            return None
        return latest


class MultiVersionStore:
    """单对象、多版本、多副本的本地事件模型。"""

    def __init__(self, replica_names: Sequence[str]):
        if not replica_names:
            raise ValueError("至少需要一个副本")
        self.replicas: Dict[str, Replica] = {
            name: Replica(name) for name in replica_names
        }
        self._seq = 0
        self._snapshots: Dict[str, Snapshot] = {}
        self._snapshot_counter = 0

    # ------------------------------------------------------------------
    # 事件应用
    # ------------------------------------------------------------------
    def _next_seq(self, seq: Optional[int]) -> int:
        if seq is None:
            self._seq += 1
            return self._seq
        if seq <= 0:
            raise ValueError("写入序号必须为正整数")
        self._seq = max(self._seq, seq)
        return seq

    def _targets(self, replicas: Optional[Sequence[str]]) -> List[Replica]:
        names = list(replicas) if replicas is not None else list(self.replicas)
        unknown = [n for n in names if n not in self.replicas]
        if unknown:
            raise KeyError("未知副本: %s" % ", ".join(unknown))
        return [self.replicas[n] for n in names]

    def _apply(self, version: Version, targets: List[Replica]) -> Version:
        for replica in targets:
            replica.versions[version.version_id] = version
        return version

    # ------------------------------------------------------------------
    # 写 / 删除 / 条件写
    # ------------------------------------------------------------------
    def put(
        self,
        etag: str,
        seq: Optional[int] = None,
        replicas: Optional[Sequence[str]] = None,
        version_id: Optional[str] = None,
    ) -> Version:
        """覆盖写:产生一个新的普通版本。"""
        seq = self._next_seq(seq)
        version = Version(version_id or "v%d" % seq, etag, seq)
        return self._apply(version, self._targets(replicas))

    def delete(
        self,
        seq: Optional[int] = None,
        replicas: Optional[Sequence[str]] = None,
    ) -> Version:
        """删除:写入删除标记。当前已是删除标记时幂等返回。"""
        targets = self._targets(replicas)
        current = targets[0].current()
        if current is not None and current.delete_marker:
            for replica in targets[1:]:
                replica.versions.setdefault(current.version_id, current)
            return current
        seq = self._next_seq(seq)
        marker = Version("v%d" % seq, etag="", seq=seq, delete_marker=True)
        return self._apply(marker, targets)

    def put_if(
        self,
        base_version_id: str,
        etag: str,
        seq: Optional[int] = None,
        replicas: Optional[Sequence[str]] = None,
    ) -> Version:
        """条件写:必须绑定指定版本。

        仅当协调副本(目标列表第一个)的当前版本就是
        base_version_id 时才写入,否则抛出 ConditionalWriteFailed。
        """
        targets = self._targets(replicas)
        current = targets[0].current()
        current_id = current.version_id if current else None
        if current_id != base_version_id:
            raise ConditionalWriteFailed(
                "条件写失败: 期望基版本 %s, 当前版本 %s"
                % (base_version_id, current_id)
            )
        return self.put(etag, seq=seq, replicas=replicas)

    # ------------------------------------------------------------------
    # 快照与读取
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        """固定当前写入序号水位,供快照读与清理保护使用。"""
        self._snapshot_counter += 1
        snap = Snapshot("snap%d" % self._snapshot_counter, self._seq)
        self._snapshots[snap.snapshot_id] = snap
        return snap

    def release(self, snapshot: Snapshot) -> None:
        self._snapshots.pop(snapshot.snapshot_id, None)

    def read(self, replica: str, snapshot: Optional[Snapshot] = None) -> Optional[Version]:
        """在指定副本上读取可见版本;快照读受快照序号约束。"""
        target = self._targets([replica])[0]
        as_of = snapshot.seq if snapshot is not None else None
        return target.visible(as_of)

    # ------------------------------------------------------------------
    # 后台清理
    # ------------------------------------------------------------------
    def _pinned_by_snapshot(self, replica: Replica) -> set:
        pinned = set()
        for snap in self._snapshots.values():
            visible = replica.visible(snap.seq)
            if visible is not None:
                pinned.add(visible.version_id)
        return pinned

    def cleanup(self, replica: str, watermark: int) -> List[str]:
        """清理指定副本上序号 <= 水位的旧版本。

        当前版本(含当前删除标记)与被活跃快照引用的版本
        不会被移除。返回被移除的版本 id 列表。
        """
        target = self._targets([replica])[0]
        target.watermark = max(target.watermark, watermark)
        current = target.current()
        pinned = self._pinned_by_snapshot(target)
        removed = []
        for version_id, version in list(target.versions.items()):
            if version.seq > watermark:
                continue
            if current is not None and version_id == current.version_id:
                continue
            if version_id in pinned:
                continue
            del target.versions[version_id]
            removed.append(version_id)
        return sorted(removed)

    # ------------------------------------------------------------------
    # 最终目录状态
    # ------------------------------------------------------------------
    def directory_state(self) -> Dict[str, dict]:
        """并发操作后的最终目录状态(按副本)。"""
        state = {}
        for name, replica in self.replicas.items():
            current = replica.current()
            state[name] = {
                "current": current.version_id if current else None,
                "watermark": replica.watermark,
                "versions": {
                    vid: {
                        "etag": v.etag,
                        "seq": v.seq,
                        "delete_marker": v.delete_marker,
                    }
                    for vid, v in sorted(
                        replica.versions.items(), key=lambda kv: kv[1].seq
                    )
                },
            }
        return state
