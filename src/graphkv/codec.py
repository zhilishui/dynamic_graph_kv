"""Optional codec for KVCOMM's Hugging Face DynamicCache state."""

from __future__ import annotations

from io import BytesIO
from typing import Any


class MissingGpuDependencies(RuntimeError):
    """Raised when the optional Torch/Transformers dependencies are absent."""


def _dependencies() -> tuple[Any, Any]:
    try:
        import torch
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:
        raise MissingGpuDependencies(
            "install GraphKV with the 'gpu' extra to use DynamicCacheCodec"
        ) from exc
    return torch, DynamicCache


def _cache_layers(cache: Any) -> list[tuple[Any, Any]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


class DynamicCacheCodec:
    """Serialize nested KVCOMM state containing tensors and DynamicCache objects.

    The wire payload uses ``torch.save`` and must only be loaded from a trusted
    GraphKV server. DynamicCache objects are first reduced to tensor-only layer
    lists so the wire format does not depend on private Transformers classes.
    """

    FORMAT_VERSION = 1

    def encode(self, state: Any) -> bytes:
        torch, DynamicCache = _dependencies()
        packed = {
            "format": self.FORMAT_VERSION,
            "state": self._pack(state, torch, DynamicCache),
        }
        buffer = BytesIO()
        torch.save(packed, buffer)
        return buffer.getvalue()

    def decode(self, payload: bytes, device: str = "cpu") -> Any:
        torch, DynamicCache = _dependencies()
        packed = torch.load(BytesIO(payload), map_location=device, weights_only=False)
        if packed.get("format") != self.FORMAT_VERSION:
            raise ValueError("unsupported GraphKV state format")
        return self._unpack(packed["state"], torch, DynamicCache, device)

    def _pack(self, value: Any, torch: Any, dynamic_cache_type: Any) -> Any:
        if isinstance(value, dynamic_cache_type):
            return {
                "__graphkv_kind__": "dynamic_cache",
                "layers": [
                    (key.detach().cpu().contiguous(), val.detach().cpu().contiguous())
                    for key, val in _cache_layers(value)
                ],
            }
        if isinstance(value, torch.Tensor):
            return {
                "__graphkv_kind__": "tensor",
                "value": value.detach().cpu().contiguous(),
            }
        if isinstance(value, dict):
            return {
                key: self._pack(val, torch, dynamic_cache_type)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._pack(item, torch, dynamic_cache_type) for item in value]
        if isinstance(value, tuple):
            return {
                "__graphkv_kind__": "tuple",
                "items": [
                    self._pack(item, torch, dynamic_cache_type) for item in value
                ],
            }
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"unsupported state value: {type(value).__name__}")

    def _unpack(
        self,
        value: Any,
        torch: Any,
        dynamic_cache_type: Any,
        device: str,
    ) -> Any:
        if isinstance(value, list):
            return [
                self._unpack(item, torch, dynamic_cache_type, device) for item in value
            ]
        if not isinstance(value, dict):
            return value
        kind = value.get("__graphkv_kind__")
        if kind == "tensor":
            return value["value"].to(device)
        if kind == "tuple":
            return tuple(
                self._unpack(item, torch, dynamic_cache_type, device)
                for item in value["items"]
            )
        if kind == "dynamic_cache":
            cache = dynamic_cache_type()
            for layer_idx, (key, val) in enumerate(value["layers"]):
                cache.update(key.to(device), val.to(device), layer_idx)
            return cache
        return {
            key: self._unpack(val, torch, dynamic_cache_type, device)
            for key, val in value.items()
        }
