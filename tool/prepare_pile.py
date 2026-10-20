#!/usr/bin/env python3
"""Download the original tokenized deduplicated Pile and randomly partition documents.

Standalone on purpose: this file can run in a staging job outside the repository.
Source format: EleutherAI/gpt-neox, megatron/data/indexed_dataset.py (MMIDIDX v1).
Source data: https://huggingface.co/datasets/EleutherAI/pythia_deduped_pile_idxmaps
This uses the original ~207B-token documents, NOT the ~300B-token preshuffled stream.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("prepare_pile")
REPO_ID = "EleutherAI/pythia_deduped_pile_idxmaps"
SOURCE_PREFIX = "pile_0.87_deduped_text_document"
FORMAT_VERSION = 1


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_index(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map uint16 MMIDIDX v1 arrays, including true document boundaries.

    An indexed item is not necessarily a document: doc_idx indexes the item array.
    See https://github.com/EleutherAI/gpt-neox/blob/v1.0/megatron/data/indexed_dataset.py
    """
    with path.open("rb") as stream:
        if stream.read(9) != b"MMIDIDX\x00\x00":
            raise ValueError("Expected GPT-NeoX MMIDIDX index")
        if struct.unpack("<Q", stream.read(8))[0] != 1:
            raise ValueError("Unsupported MMIDIDX version")
        if stream.read(1) != b"\x08":
            raise ValueError("Expected uint16 token IDs (MMIDIDX dtype code 8)")
        items, documents = struct.unpack("<QQ", stream.read(16))
    if items == 0 or documents < 3:
        raise ValueError("Need at least two documents")
    if path.stat().st_size != 34 + items * 12 + documents * 8:
        raise ValueError("Truncated or extended MMIDIDX index")
    sizes = np.memmap(path, dtype="<i4", mode="r", offset=34, shape=(items,))
    pointers = np.memmap(
        path, dtype="<i8", mode="r", offset=34 + 4 * items, shape=(items,)
    )
    doc_idx = np.memmap(
        path, dtype="<i8", mode="r", offset=34 + 12 * items, shape=(documents,)
    )
    if doc_idx[0] != 0 or doc_idx[-1] != items or pointers[0] != 0:
        raise ValueError("Invalid document/item boundary endpoints")
    for start in range(0, items, 1_000_000):
        end = min(items, start + 1_000_000)
        if np.any(sizes[start:end] < 0) or np.any(pointers[start:end] % 2):
            raise ValueError("Negative lengths or unaligned pointers")
        if end > start + 1 and np.any(
            np.diff(pointers[start:end]) != sizes[start : end - 1].astype(np.int64) * 2
        ):
            raise ValueError("Source items must form a contiguous token stream")
        if end < items and pointers[end] != pointers[end - 1] + int(sizes[end - 1]) * 2:
            raise ValueError("Noncontiguous source item boundary")
    for start in range(0, documents - 1, 1_000_000):
        part = doc_idx[start : min(documents, start + 1_000_001)]
        if np.any(part < 0) or np.any(part > items) or np.any(np.diff(part) < 0):
            raise ValueError("Invalid document index")
    return sizes, pointers, doc_idx


def source_files(directory: Path) -> list[Path]:
    merged = directory / f"{SOURCE_PREFIX}.bin"
    if merged.is_file():
        return [merged]
    shards = sorted(directory.glob(f"{SOURCE_PREFIX}-*-of-*.bin"))
    if not shards:
        raise FileNotFoundError(f"No source .bin files in {directory}")
    for number, path in enumerate(shards):
        if path.name != f"{SOURCE_PREFIX}-{number:05d}-of-{len(shards) - 1:05d}.bin":
            raise ValueError("Missing, extra, or incorrectly named source shard")
    return shards


class ShardReader:
    """Sequential virtual concatenation; never write a second merged source copy."""

    def __init__(self, files: list[Path], offset: int):
        self.files = files
        self.ends = np.cumsum([p.stat().st_size for p in files], dtype=np.int64)
        self.position = offset
        self.stream = None
        self.number = -1

    def read(self, count: int) -> bytes:
        chunks = []
        while count:
            number = int(np.searchsorted(self.ends, self.position, side="right"))
            if number >= len(self.files):
                raise EOFError("Truncated source token stream")
            if number != self.number:
                if self.stream is not None:
                    self.stream.close()
                self.number = number
                self.stream = self.files[number].open("rb")
                base = 0 if number == 0 else int(self.ends[number - 1])
                self.stream.seek(self.position - base)
            take = min(count, int(self.ends[number]) - self.position)
            block = self.stream.read(take)
            if len(block) != take:
                raise EOFError("Truncated source shard")
            chunks.append(block)
            self.position += take
            count -= take
        return b"".join(chunks)

    def close(self):
        if self.stream is not None:
            self.stream.close()


def prepare(
    source: Path,
    output: Path,
    *,
    seed: int = 42,
    block_tokens: int = 32 * 1024**2,
    vocabulary_size: int = 50304,
    stop_after_blocks: int | None = None,
) -> dict | None:
    """Random exact 50/50 document membership, sequential I/O, resumable token copying.

    PCG64 permutation makes every equal-size document partition equally likely.
    Within each output, source order is retained to avoid random disk reads.
    No token, EOS, or padding is added/removed. Empty source documents are retained.
    """
    if seed < 0 or block_tokens <= 0:
        raise ValueError("Seed must be nonnegative and block size positive")
    output.mkdir(parents=True, exist_ok=True)
    sizes, pointers, documents = read_index(source / f"{SOURCE_PREFIX}.idx")
    files = source_files(source)
    total_bytes = sum(p.stat().st_size for p in files)
    expected_bytes = int(pointers[-1]) + int(sizes[-1]) * 2
    if total_bytes != expected_bytes:
        raise ValueError(f"Source size mismatch: {total_bytes} != {expected_bytes}")
    tokens = total_bytes // 2
    count = len(documents) - 1
    download_manifest = source / "download_manifest.json"
    identity = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "documents": count,
        "tokens": tokens,
        "source_index_sha256": sha256(source / f"{SOURCE_PREFIX}.idx"),
        "vocabulary_size": vocabulary_size,
    }
    if download_manifest.exists():
        identity["download_manifest_sha256"] = sha256(download_manifest)
    identity_file = output / "identity.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError(
                "Existing output belongs to a different source, seed, or configuration"
            )
    else:
        if any(output.iterdir()):
            raise ValueError(
                "Refusing to use a nonempty directory without identity.json"
            )
        atomic_json(identity_file, identity)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        for details in manifest["files"].values():
            if (output / details["path"]).stat().st_size != details["bytes"]:
                raise ValueError("Completed output file size changed")
        LOGGER.info("Completed dataset already exists: %s", output)
        return manifest

    assignment_path = output / "document_partition.npy"
    if not assignment_path.exists():
        rng = np.random.Generator(np.random.PCG64(seed))
        order = rng.permutation(count)
        assignment = np.ones(count, dtype=np.uint8)
        assignment[order[: count // 2]] = 0
        temporary = assignment_path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, assignment, allow_pickle=False)
        os.replace(temporary, assignment_path)
        del assignment, order
    assignment = np.load(assignment_path, mmap_mode="r", allow_pickle=False)
    offsets_path = output / "source.doc_offsets.npy"
    if not offsets_path.exists():
        temporary = offsets_path.with_suffix(".tmp")
        offsets = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="<u8", shape=(count + 1,)
        )
        for start in range(0, count, 1_000_000):
            end = min(count, start + 1_000_000)
            indices = documents[start:end]
            offsets[start:end] = np.where(
                indices < len(pointers),
                pointers[np.minimum(indices, len(pointers) - 1)] // 2,
                tokens,
            )
        offsets[-1] = tokens
        offsets.flush()
        del offsets
        os.replace(temporary, offsets_path)
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if assignment.shape != (count,) or offsets.shape != (count + 1,):
        raise ValueError("Invalid cached partition metadata")

    partitions = {}
    for value, name in enumerate(("A", "B")):
        selected_count = int(np.count_nonzero(assignment == value))
        if selected_count == 0:
            raise ValueError("Empty document partition")
        ids_path = output / f"{name}.source_document_ids.npy"
        part_offsets_path = output / f"{name}.doc_offsets.npy"
        if not ids_path.exists() or not part_offsets_path.exists():
            ids = np.lib.format.open_memmap(
                ids_path.with_suffix(".tmp"),
                mode="w+",
                dtype="<u8",
                shape=(selected_count,),
            )
            part_offsets = np.lib.format.open_memmap(
                part_offsets_path.with_suffix(".tmp"),
                mode="w+",
                dtype="<u8",
                shape=(selected_count + 1,),
            )
            cursor = token_cursor = 0
            part_offsets[0] = 0
            for start in range(0, count, 1_000_000):
                end = min(count, start + 1_000_000)
                selected = np.flatnonzero(assignment[start:end] == value) + start
                lengths = offsets[selected + 1] - offsets[selected]
                ids[cursor : cursor + len(selected)] = selected
                part_offsets[cursor + 1 : cursor + len(selected) + 1] = (
                    token_cursor + np.cumsum(lengths, dtype=np.uint64)
                )
                token_cursor += int(lengths.sum(dtype=np.uint64))
                cursor += len(selected)
            ids.flush()
            part_offsets.flush()
            del ids, part_offsets
            os.replace(ids_path.with_suffix(".tmp"), ids_path)
            os.replace(part_offsets_path.with_suffix(".tmp"), part_offsets_path)
        part_offsets = np.load(part_offsets_path, mmap_mode="r", allow_pickle=False)
        partitions[name] = {
            "documents": selected_count,
            "tokens": int(part_offsets[-1]),
            "data": f"{name}.bin",
            "doc_offsets": part_offsets_path.name,
            "source_document_ids": ids_path.name,
        }
    if sum(p["tokens"] for p in partitions.values()) != tokens:
        raise ValueError("Partition metadata loses or duplicates source tokens")

    state_path = output / "copy_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"cursor": 0, "A": 0, "B": 0}
    )
    remaining = total_bytes - 2 * (state["A"] + state["B"])
    if shutil.disk_usage(output).free < remaining + 1024**3:
        raise OSError("Insufficient space for remaining A/B data plus 1 GiB margin")
    reader = ShardReader(files, 2 * state["cursor"])
    try:
        with contextlib.ExitStack() as stack:
            streams = {}
            for name in ("A", "B"):
                path = output / f"{name}.bin"
                stream = stack.enter_context(
                    path.open("r+b" if path.exists() else "w+b")
                )
                committed_bytes = 2 * state[name]
                if path.stat().st_size < committed_bytes:
                    raise ValueError("Output is shorter than its committed copy state")
                stream.truncate(committed_bytes)
                stream.seek(committed_bytes)
                streams[name] = stream
            blocks = 0
            last_log = last_commit = time.monotonic()
            while state["cursor"] < tokens:
                begin = state["cursor"]
                end = min(tokens, begin + block_tokens)
                block = np.frombuffer(reader.read((end - begin) * 2), dtype="<u2")
                if block.size and int(block.max()) >= vocabulary_size:
                    raise ValueError("Source token ID outside tokenizer vocabulary")
                first = int(np.searchsorted(offsets, begin, side="right")) - 1
                last = int(np.searchsorted(offsets, end, side="left"))
                bounds = np.clip(offsets[first : last + 1], begin, end)
                labels = np.repeat(
                    assignment[first:last], np.diff(bounds).astype(np.int64)
                )
                if len(labels) != len(block):
                    raise ValueError("Document routing does not cover the token block")
                for value, name in enumerate(("A", "B")):
                    routed = block[labels == value]
                    streams[name].write(routed.tobytes())
                    state[name] += len(routed)
                state["cursor"] = end
                blocks += 1
                now = time.monotonic()
                if now - last_log >= 15 or end == tokens:
                    LOGGER.info(
                        "Copy %.2f%%: %.3fB / %.3fB tokens",
                        100 * end / tokens,
                        end / 1e9,
                        tokens / 1e9,
                    )
                    last_log = now
                if (
                    now - last_commit >= 60
                    or end == tokens
                    or stop_after_blocks == blocks
                ):
                    for stream in streams.values():
                        stream.flush()
                        os.fsync(stream.fileno())
                    atomic_json(state_path, state)
                    last_commit = now
                if stop_after_blocks == blocks:
                    return None
    finally:
        reader.close()
    for name in ("A", "B"):
        if state[name] != partitions[name]["tokens"]:
            raise ValueError(
                "Copied token counts differ from document partition metadata"
            )
    LOGGER.info("Hashing completed output files")
    details = {}
    for path in sorted(output.iterdir()):
        if path.suffix in (".bin", ".npy") and path.name != "source.doc_offsets.npy":
            details[path.name] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = dict(
        **identity,
        dtype="uint16",
        byte_order="little",
        partitions=partitions,
        files=details,
        split_method="PCG64(seed).permutation(documents); first floor(N/2) -> A, remainder -> B",
        within_partition_order="original source document order",
        source_repo=REPO_ID,
        tokenizer="EleutherAI/pythia-14m",
        eos_token_id=0,
    )
    manifest["numpy_version"] = np.__version__
    if download_manifest.exists():
        manifest["source_download"] = json.loads(download_manifest.read_text())
    assets = source.parent / "tokenizer" / "assets_manifest.json"
    if assets.exists():
        manifest["tokenizer_assets"] = json.loads(assets.read_text())
    atomic_json(output / "manifest.json", manifest)
    LOGGER.info("Complete: %s", json.dumps(partitions, sort_keys=True))
    return manifest


def download(
    directory: Path, *, revision: str, workers: int, plan_only: bool = False
) -> None:
    # Keep the potentially large HF/Xet caches out of SURF's 200-GiB home quota.
    os.environ.setdefault("HF_HOME", str(directory / ".hf_cache"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
    from huggingface_hub import HfApi, snapshot_download

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "download_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if revision not in ("main", previous["revision"]):
            raise ValueError("Use a new source directory for a different revision")
        revision = previous["revision"]
    info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
    selected = [
        f
        for f in info.siblings
        if f.rfilename.startswith(SOURCE_PREFIX)
        and f.rfilename.endswith((".bin", ".idx"))
    ]
    if not selected or len([f for f in selected if f.rfilename.endswith(".idx")]) != 1:
        raise ValueError("Unexpected source repository file layout")
    records = [
        {
            "path": f.rfilename,
            "bytes": f.size,
            "sha256": f.lfs.sha256 if f.lfs else None,
        }
        for f in selected
    ]
    if any(f["sha256"] is None for f in records):
        raise ValueError("Source data must expose upstream SHA256 hashes")
    LOGGER.info(
        "Pinned source revision %s; %d files; %.2f GB",
        info.sha,
        len(records),
        sum(f["bytes"] for f in records) / 1e9,
    )
    if plan_only:
        return
    missing = sum(
        max(
            0,
            f["bytes"]
            - (
                (directory / f["path"]).stat().st_size
                if (directory / f["path"]).exists()
                else 0
            ),
        )
        for f in records
    )
    if shutil.disk_usage(directory).free < missing + 1024**3:
        raise OSError("Insufficient space for source download")
    atomic_json(
        manifest_path, {"repo_id": REPO_ID, "revision": info.sha, "files": records}
    )
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=info.sha,
        local_dir=directory,
        allow_patterns=[f["path"] for f in records],
        max_workers=workers,
    )
    verified_path = directory / "verified_files.json"
    verified = json.loads(verified_path.read_text()) if verified_path.exists() else {}

    def verify(record):
        path = directory / record["path"]
        stamp = {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": record["sha256"],
        }
        if stamp["bytes"] != record["bytes"]:
            raise ValueError(f"Downloaded size mismatch: {path}")
        if verified.get(record["path"]) != stamp and sha256(path) != record["sha256"]:
            raise ValueError(
                f"Downloaded SHA256 mismatch: {path}; remove the corrupt file and retry"
            )
        return record["path"], stamp

    LOGGER.info("Verifying upstream SHA256 hashes (cached on unchanged files)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        for name, stamp in executor.map(verify, records):
            verified[name] = stamp
            atomic_json(verified_path, verified)
    # Only configuration/tokenizer assets; NEVER download pretrained weights.
    tokenizer_dir = directory.parent / "tokenizer"
    tokenizer_manifest = tokenizer_dir / "assets_manifest.json"
    model_revision = (
        json.loads(tokenizer_manifest.read_text())["revision"]
        if tokenizer_manifest.exists()
        else "main"
    )
    model_info = HfApi().model_info("EleutherAI/pythia-14m", revision=model_revision)
    snapshot_download(
        "EleutherAI/pythia-14m",
        revision=model_info.sha,
        local_dir=tokenizer_dir,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        max_workers=workers,
    )
    atomic_json(
        tokenizer_manifest,
        {
            "repo_id": "EleutherAI/pythia-14m",
            "revision": model_info.sha,
            "pretrained_weights_downloaded": False,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing original MMIDIDX files; skip downloads",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--block-tokens", type=int, default=32 * 1024**2)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Fetch metadata only; do not download or split data",
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.workers <= 0 or args.block_tokens <= 0:
        parser.error(
            "seed must be nonnegative; workers and block-tokens must be positive"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.root.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent jobs from publishing inconsistent files to the same root.
    import fcntl

    with (args.root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = args.source_dir or args.root / "source"
        if args.source_dir is None:
            download(
                source,
                revision=args.revision,
                workers=args.workers,
                plan_only=args.plan_only,
            )
        if not args.plan_only:
            prepare(
                source,
                args.root / f"ab_seed{args.seed}",
                seed=args.seed,
                block_tokens=args.block_tokens,
            )


if __name__ == "__main__":
    main()
