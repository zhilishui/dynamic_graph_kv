import threading
import unittest

from graphkv.server import StateTcpServer
from graphkv.store import MemoryStateStore, TcpStateStoreClient, VersionConflict


class MemoryStoreTests(unittest.TestCase):
    def test_versions_and_compare_and_swap(self) -> None:
        store = MemoryStateStore(max_bytes=32)
        first = store.put("key", b"one", expected_version=0)
        second = store.put("key", b"two", expected_version=first.version)
        self.assertEqual(second.version, 2)
        with self.assertRaises(VersionConflict):
            store.put("key", b"stale", expected_version=1)

    def test_lru_evicts_old_record(self) -> None:
        store = MemoryStateStore(max_bytes=6)
        store.put("a", b"aaa")
        store.put("b", b"bbb")
        store.get("a")
        store.put("c", b"ccc")
        self.assertIsNotNone(store.get("a"))
        self.assertIsNone(store.get("b"))
        self.assertEqual(store.stats()["evictions"], 1)


class TcpStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = StateTcpServer(
            ("127.0.0.1", 0), MemoryStateStore(max_bytes=1 << 20)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = TcpStateStoreClient(host, port)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_round_trip_and_metadata(self) -> None:
        self.assertTrue(self.client.ping())
        written = self.client.put("layout:1", b"kv-bytes", {"role": "C"})
        loaded = self.client.get("layout:1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.payload, b"kv-bytes")
        self.assertEqual(loaded.metadata, {"role": "C"})
        self.assertEqual(loaded.version, written.version)
        self.assertEqual(self.client.stats()["entries"], 1)

    def test_remote_version_conflict(self) -> None:
        self.client.put("layout:1", b"one", expected_version=0)
        with self.assertRaises(VersionConflict):
            self.client.put("layout:1", b"two", expected_version=0)


if __name__ == "__main__":
    unittest.main()
