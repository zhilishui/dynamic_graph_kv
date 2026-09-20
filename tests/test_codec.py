import importlib.util
import unittest

HAS_GPU_DEPS = bool(importlib.util.find_spec("torch")) and bool(
    importlib.util.find_spec("transformers")
)


@unittest.skipUnless(HAS_GPU_DEPS, "optional torch/transformers dependencies absent")
class CodecTests(unittest.TestCase):
    def test_dynamic_cache_round_trip_on_cpu(self) -> None:
        import torch
        from transformers.cache_utils import DynamicCache

        from graphkv.codec import DynamicCacheCodec

        cache = DynamicCache()
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        value = key + 100
        cache.update(key, value, 0)
        codec = DynamicCacheCodec()
        restored = codec.decode(codec.encode({"cache": cache}))
        restored_cache = restored["cache"]
        if hasattr(restored_cache, "layers"):
            restored_key = restored_cache.layers[0].keys
            restored_value = restored_cache.layers[0].values
        else:
            restored_key = restored_cache.key_cache[0]
            restored_value = restored_cache.value_cache[0]
        self.assertTrue(torch.equal(restored_key, key))
        self.assertTrue(torch.equal(restored_value, value))


if __name__ == "__main__":
    unittest.main()
