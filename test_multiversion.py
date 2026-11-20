"""多版本并发本地事件模型的单元测试。

覆盖:并发覆盖、重复删除、旧条件写、多副本差异,
以及删除标记可见性与清理水位/快照保护边界。
"""

import unittest

from multiversion import (
    ConditionalWriteFailed,
    MultiVersionStore,
)


class ConcurrentOverwriteTest(unittest.TestCase):
    def test_last_writer_wins_and_old_version_hidden(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")          # v1
        store.put("etag-b")          # v2, 并发覆盖后序号更大者胜出
        visible = store.read("r1")
        self.assertEqual(visible.version_id, "v2")
        self.assertEqual(visible.etag, "etag-b")
        # 普通旧版本不能再被当前读看到
        self.assertNotEqual(visible.version_id, "v1")

    def test_out_of_order_arrival_uses_seq_not_arrival_order(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-new", seq=2)  # 高序号事件先到达
        store.put("etag-old", seq=1)  # 低序号事件后到达
        visible = store.read("r1")
        self.assertEqual(visible.etag, "etag-new")
        self.assertEqual(visible.seq, 2)


class DuplicateDeleteTest(unittest.TestCase):
    def test_delete_marker_hides_current_read(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")
        store.delete()
        self.assertIsNone(store.read("r1"))

    def test_duplicate_delete_is_idempotent(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")
        first = store.delete()
        second = store.delete()
        self.assertEqual(first.version_id, second.version_id)
        versions = store.directory_state()["r1"]["versions"]
        self.assertEqual(len(versions), 2)  # 普通版本 + 一个删除标记
        markers = [v for v in versions.values() if v["delete_marker"]]
        self.assertEqual(len(markers), 1)


class StaleConditionalWriteTest(unittest.TestCase):
    def test_conditional_write_binds_to_specified_version(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")           # v1
        store.put("etag-b")           # v2, 并发覆盖
        # 绑定旧版本 v1 的条件写必须失败
        with self.assertRaises(ConditionalWriteFailed):
            store.put_if("v1", "etag-c")
        # 绑定当前版本 v2 的条件写成功
        v3 = store.put_if("v2", "etag-c")
        self.assertEqual(v3.version_id, "v3")
        self.assertEqual(store.read("r1").etag, "etag-c")

    def test_conditional_write_on_empty_store_requires_no_version(self):
        store = MultiVersionStore(["r1"])
        with self.assertRaises(ConditionalWriteFailed):
            store.put_if("v1", "etag-a")


class MultiReplicaDivergenceTest(unittest.TestCase):
    def test_replica_without_event_sees_stale_state(self):
        store = MultiVersionStore(["r1", "r2"])
        store.put("etag-a")                        # 两个副本都应用
        store.put("etag-b", replicas=["r1"])       # 只到达 r1
        # 某个版本在 r1 可见并不代表 r2 看到相同状态
        self.assertEqual(store.read("r1").etag, "etag-b")
        self.assertEqual(store.read("r2").etag, "etag-a")
        # 事件传播后两副本收敛
        store.put("etag-b", seq=2, version_id="v2", replicas=["r2"])
        self.assertEqual(store.read("r2").etag, "etag-b")

    def test_final_directory_state_per_replica(self):
        store = MultiVersionStore(["r1", "r2"])
        store.put("etag-a", replicas=["r1"])
        state = store.directory_state()
        self.assertEqual(state["r1"]["current"], "v1")
        self.assertIsNone(state["r2"]["current"])
        self.assertEqual(state["r2"]["versions"], {})


class CleanupTest(unittest.TestCase):
    def test_cleanup_removes_old_versions_above_nothing_pinned(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")  # v1
        store.put("etag-b")  # v2
        store.put("etag-c")  # v3
        removed = store.cleanup("r1", watermark=2)
        self.assertEqual(removed, ["v1", "v2"])
        state = store.directory_state()["r1"]
        self.assertEqual(list(state["versions"]), ["v3"])
        self.assertEqual(state["watermark"], 2)

    def test_cleanup_keeps_versions_newer_than_watermark(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")  # v1
        store.put("etag-b")  # v2
        removed = store.cleanup("r1", watermark=0)
        self.assertEqual(removed, [])
        self.assertEqual(len(store.directory_state()["r1"]["versions"]), 2)

    def test_cleanup_cannot_remove_snapshot_referenced_version(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")          # v1
        snap = store.snapshot()      # 快照固定在 seq=1
        store.put("etag-b")          # v2
        store.put("etag-c")          # v3
        removed = store.cleanup("r1", watermark=3)
        # v1 被快照引用,不能移除;v2 无引用可移除;v3 是当前版本
        self.assertEqual(removed, ["v2"])
        versions = store.directory_state()["r1"]["versions"]
        self.assertEqual(sorted(versions), ["v1", "v3"])
        # 快照读仍能看到 v1
        self.assertEqual(store.read("r1", snapshot=snap).etag, "etag-a")
        # 释放快照后 v1 才可被清理
        store.release(snap)
        removed = store.cleanup("r1", watermark=3)
        self.assertEqual(removed, ["v1"])

    def test_cleanup_keeps_current_delete_marker(self):
        store = MultiVersionStore(["r1"])
        store.put("etag-a")  # v1
        store.delete()       # v2 删除标记(当前版本)
        removed = store.cleanup("r1", watermark=10)
        self.assertEqual(removed, ["v1"])
        state = store.directory_state()["r1"]
        self.assertEqual(state["current"], "v2")
        self.assertTrue(state["versions"]["v2"]["delete_marker"])
        self.assertIsNone(store.read("r1"))


if __name__ == "__main__":
    unittest.main()
