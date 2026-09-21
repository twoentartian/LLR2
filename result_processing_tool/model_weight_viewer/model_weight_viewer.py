"""Local web viewer for raw values stored in PyTorch model checkpoints.

The server never instantiates the saved model. It reads the checkpoint on CPU
with ``weights_only=True`` and exposes tensor metadata and bounded pages of raw
values to the bundled browser UI.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import torch


ASSET_DIR = Path(__file__).resolve().parent
STATIC_FILES = {
    "/": ("model_weight_viewer.html", "text/html; charset=utf-8"),
    "/model_weight_viewer.css": ("model_weight_viewer.css", "text/css; charset=utf-8"),
    "/model_weight_viewer.js": ("model_weight_viewer.js", "text/javascript; charset=utf-8"),
}
MAX_PAGE_SIZE = 5_000
DEFAULT_PORT = 43817


@dataclass
class LoadedCheckpoint:
    filename: str
    model_type: str | None
    dataset_type: str | None
    source_format: str
    state_dict: dict[str, Any]
    tensors: list[dict[str, Any]]
    loaded_at: float = field(default_factory=time.time)

    @property
    def tensor_by_name(self) -> dict[str, torch.Tensor]:
        return {
            name: value for name, value in self.state_dict.items()
            if torch.is_tensor(value)
        }


def _metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _extract_state_dict(payload: Any) -> tuple[dict[str, Any], str | None, str | None, str]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint root must be a mapping, got {type(payload).__name__}")

    model_type = _metadata_text(payload.get("model_name"))
    dataset_type = _metadata_text(payload.get("dataset_name"))
    for key, label in (("state_dict", "LLR2 state_dict"),
                       ("model_state_dict", "model_state_dict")):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate), model_type, dataset_type, label

    # A plain PyTorch state_dict is itself a mapping from names to tensors.
    if payload and all(isinstance(key, str) for key in payload):
        return dict(payload), model_type, dataset_type, "raw state_dict"
    if not payload:
        return {}, model_type, dataset_type, "raw state_dict"
    raise ValueError("checkpoint does not contain a state_dict mapping")


def _tensor_summary(name: str, value: Any) -> dict[str, Any]:
    if not torch.is_tensor(value):
        preview = repr(value)
        return {
            "name": name,
            "kind": "object",
            "python_type": type(value).__name__,
            "preview": preview[:240] + ("…" if len(preview) > 240 else ""),
        }

    dtype = str(value.dtype).removeprefix("torch.")
    layout = str(value.layout).removeprefix("torch.")
    summary = {
        "name": name,
        "kind": "tensor",
        "shape": list(value.shape),
        "ndim": value.ndim,
        "dtype": dtype,
        "layout": layout,
        "numel": value.numel(),
        "bytes": value.numel() * value.element_size(),
        "requires_grad": bool(value.requires_grad),
    }
    if value.is_quantized:
        summary["quantized"] = True
        summary["qscheme"] = str(value.qscheme()).removeprefix("torch.")
        if value.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            summary["scale"] = value.q_scale()
            summary["zero_point"] = value.q_zero_point()
    return summary


def load_checkpoint_file(path: Path, filename: str | None = None) -> LoadedCheckpoint:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"PyTorch could not read this checkpoint safely: {exc}") from exc
    state_dict, model_type, dataset_type, source_format = _extract_state_dict(payload)
    tensors = [_tensor_summary(str(name), value) for name, value in state_dict.items()]
    return LoadedCheckpoint(
        filename=filename or path.name,
        model_type=model_type,
        dataset_type=dataset_type,
        source_format=source_format,
        state_dict={str(name): value for name, value in state_dict.items()},
        tensors=tensors,
    )


def _readable_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "meta":
        raise ValueError("meta tensors do not contain readable values")
    value = value.detach().cpu()
    if value.layout != torch.strided:
        value = value.to_dense()
    if value.is_quantized:
        value = value.dequantize()
    return value


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    return str(value)


def tensor_value_page(value: torch.Tensor, offset: int, limit: int) -> dict[str, Any]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
    readable = _readable_tensor(value)
    total = readable.numel()
    offset = min(offset, total)
    end = min(offset + limit, total)
    flat = readable.reshape(-1)
    values = [_scalar_text(item) for item in flat[offset:end].tolist()]
    return {
        "offset": offset,
        "limit": limit,
        "returned": len(values),
        "total": total,
        "values": values,
    }


def tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
    readable = _readable_tensor(value)
    if readable.numel() == 0:
        return {"count": 0}
    if readable.is_complex():
        readable = readable.abs()
        value_kind = "magnitude"
    else:
        value_kind = "value"
    numeric = readable.to(torch.float64).reshape(-1)
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    result: dict[str, Any] = {
        "count": numeric.numel(),
        "finite_count": finite_values.numel(),
        "nan_count": int(torch.isnan(numeric).sum().item()),
        "positive_infinity_count": int(torch.isposinf(numeric).sum().item()),
        "negative_infinity_count": int(torch.isneginf(numeric).sum().item()),
        "zero_count": int((numeric == 0).sum().item()),
        "value_kind": value_kind,
    }
    if finite_values.numel():
        result.update({
            "minimum": _scalar_text(finite_values.min().item()),
            "maximum": _scalar_text(finite_values.max().item()),
            "mean": _scalar_text(finite_values.mean().item()),
            "standard_deviation": _scalar_text(
                finite_values.std(unbiased=False).item(),
            ),
        })
    return result


def checkpoint_overview(checkpoint_id: str, checkpoint: LoadedCheckpoint) -> dict[str, Any]:
    tensor_entries = [item for item in checkpoint.tensors if item["kind"] == "tensor"]
    return {
        "checkpoint_id": checkpoint_id,
        "filename": checkpoint.filename,
        "model_type": checkpoint.model_type,
        "dataset_type": checkpoint.dataset_type,
        "source_format": checkpoint.source_format,
        "tensor_count": len(tensor_entries),
        "entry_count": len(checkpoint.tensors),
        "total_values": sum(item["numel"] for item in tensor_entries),
        "total_bytes": sum(item["bytes"] for item in tensor_entries),
        "entries": checkpoint.tensors,
    }


class CheckpointStore:
    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
        self._items: OrderedDict[str, LoadedCheckpoint] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, checkpoint: LoadedCheckpoint) -> str:
        checkpoint_id = uuid.uuid4().hex
        with self._lock:
            self._items[checkpoint_id] = checkpoint
            while len(self._items) > self.max_checkpoints:
                self._items.popitem(last=False)
        return checkpoint_id

    def get(self, checkpoint_id: str) -> LoadedCheckpoint:
        with self._lock:
            checkpoint = self._items.get(checkpoint_id)
            if checkpoint is None:
                raise KeyError("checkpoint is no longer loaded")
            self._items.move_to_end(checkpoint_id)
            return checkpoint

    def remove(self, checkpoint_id: str) -> None:
        with self._lock:
            self._items.pop(checkpoint_id, None)


class WeightViewerHandler(BaseHTTPRequestHandler):
    server_version = "LLR2WeightViewer/1.0"
    store = CheckpointStore()

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def _send_bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[parsed.path]
            try:
                self._send_bytes((ASSET_DIR / filename).read_bytes(), content_type)
            except FileNotFoundError:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"missing web asset: {filename}")
            return
        if parsed.path == "/api/tensor":
            self._handle_tensor_values(parse_qs(parsed.query))
            return
        if parsed.path == "/api/statistics":
            self._handle_tensor_statistics(parse_qs(parsed.query))
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/load":
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        content_length_text = self.headers.get("Content-Length")
        if content_length_text is None:
            self._send_error_json(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            return
        try:
            content_length = int(content_length_text)
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return
        if content_length <= 0:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "empty checkpoint upload")
            return

        encoded_name = self.headers.get("X-Filename", "checkpoint.model.pt")
        filename = Path(unquote(encoded_name)).name
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="llr2_weight_viewer_", suffix=".pt", delete=False) as handle:
                temporary_path = Path(handle.name)
                remaining = content_length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("upload ended before Content-Length bytes were received")
                    handle.write(chunk)
                    remaining -= len(chunk)
            checkpoint = load_checkpoint_file(temporary_path, filename=filename)
            checkpoint_id = self.store.add(checkpoint)
            self._send_json(checkpoint_overview(checkpoint_id, checkpoint))
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"failed to load checkpoint: {exc}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/checkpoint":
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        checkpoint_id = parse_qs(parsed.query).get("id", [""])[0]
        self.store.remove(checkpoint_id)
        self._send_json({"removed": True})

    def _get_tensor(self, query: dict[str, list[str]]) -> tuple[torch.Tensor, str]:
        checkpoint_id = query.get("id", [""])[0]
        name = query.get("name", [""])[0]
        checkpoint = self.store.get(checkpoint_id)
        value = checkpoint.state_dict.get(name)
        if value is None or not torch.is_tensor(value):
            raise KeyError(f"tensor {name!r} was not found")
        return value, name

    def _handle_tensor_values(self, query: dict[str, list[str]]) -> None:
        try:
            value, name = self._get_tensor(query)
            offset = int(query.get("offset", ["0"])[0])
            limit = int(query.get("limit", ["100"])[0])
            page = tensor_value_page(value, offset, limit)
            page.update({"name": name, "shape": list(value.shape),
                         "dtype": str(value.dtype).removeprefix("torch.")})
            self._send_json(page)
        except (KeyError, ValueError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_tensor_statistics(self, query: dict[str, list[str]]) -> None:
        try:
            value, name = self._get_tensor(query)
            self._send_json({"name": name, "statistics": tensor_statistics(value)})
        except (KeyError, ValueError, RuntimeError) as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Browse raw values in LLR2 .model.pt files")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--quiet", action="store_true", help="do not print the server URL")
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), WeightViewerHandler)
    if not args.quiet:
        print(f"Model weight viewer: http://{args.host}:{args.port}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
