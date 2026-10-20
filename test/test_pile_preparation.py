"""Document isolation, exact token conservation, and interrupted-copy recovery."""

import struct
from pathlib import Path

import numpy as np
import pytest

from tool.prepare_pile import SOURCE_PREFIX, prepare, read_index


def make_source(root: Path, documents, *, shard_bytes=14):
    root.mkdir()
    # Exercise true document boundaries: each document contains two indexed items.
    items = [
        np.asarray(part, dtype="<u2")
        for doc in documents
        for part in (doc[:1], doc[1:])
    ]
    sizes = np.asarray([len(x) for x in items], dtype="<i4")
    pointers = np.concatenate(([0], np.cumsum(sizes[:-1], dtype=np.int64))) * 2
    doc_idx = np.arange(0, len(items) + 1, 2, dtype="<i8")
    with (root / f"{SOURCE_PREFIX}.idx").open("wb") as stream:
        stream.write(
            b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 8, len(items), len(doc_idx))
        )
        stream.write(sizes.tobytes())
        stream.write(pointers.astype("<i8").tobytes())
        stream.write(doc_idx.tobytes())
    data = np.concatenate(items).tobytes()
    blocks = [data[i : i + shard_bytes] for i in range(0, len(data), shard_bytes)]
    for i, block in enumerate(blocks):
        (root / f"{SOURCE_PREFIX}-{i:05d}-of-{len(blocks) - 1:05d}.bin").write_bytes(
            block
        )
    return root


def assert_partition(output, documents, manifest):
    all_ids = []
    for name in ("A", "B"):
        ids = np.load(output / f"{name}.source_document_ids.npy")
        offsets = np.load(output / f"{name}.doc_offsets.npy")
        tokens = np.fromfile(output / f"{name}.bin", dtype="<u2")
        expected = np.asarray(
            [token for i in ids for token in documents[int(i)]], dtype="<u2"
        )
        np.testing.assert_array_equal(tokens, expected)
        np.testing.assert_array_equal(
            np.diff(offsets), [len(documents[int(i)]) for i in ids]
        )
        assert manifest["partitions"][name]["tokens"] == len(expected)
        all_ids.extend(ids.tolist())
    assert sorted(all_ids) == list(range(len(documents)))
    assert (
        abs(
            manifest["partitions"]["A"]["documents"]
            - manifest["partitions"]["B"]["documents"]
        )
        <= 1
    )


def test_random_document_partition_preserves_all_tokens(tmp_path):
    documents = [
        [],
        [11, 12],
        [21],
        [],
        [31, 32, 33, 34, 35, 36, 37],
        [41, 42],
        [],
        [51],
        [61, 62],
        [],
    ]
    source = make_source(tmp_path / "source", documents)
    first = prepare(source, tmp_path / "first", seed=42, block_tokens=3)
    assert_partition(tmp_path / "first", documents, first)
    second = prepare(source, tmp_path / "second", seed=42, block_tokens=8)
    different = prepare(source, tmp_path / "different", seed=7, block_tokens=4)
    for name in ("A", "B"):
        assert (tmp_path / "first" / f"{name}.bin").read_bytes() == (
            tmp_path / "second" / f"{name}.bin"
        ).read_bytes()
    assert first["files"] == second["files"]
    assert (
        first["files"]["document_partition.npy"]
        != different["files"]["document_partition.npy"]
    )
    assert prepare(source, tmp_path / "first", seed=42) == first
    with pytest.raises(ValueError, match="different source"):
        prepare(source, tmp_path / "first", seed=4)


def test_resume_truncates_uncommitted_bytes(tmp_path):
    documents = [[i, i + 1, i + 2, i + 3] for i in range(20, 100, 10)]
    source = make_source(tmp_path / "source", documents)
    output = tmp_path / "resumed"
    assert prepare(source, output, block_tokens=5, stop_after_blocks=2) is None
    assert not (output / "manifest.json").exists()
    for name in ("A", "B"):
        with (output / f"{name}.bin").open("ab") as stream:
            stream.write(b"uncommitted")
    manifest = prepare(source, output, block_tokens=3)
    reference = prepare(source, tmp_path / "reference", block_tokens=7)
    assert manifest["files"] == reference["files"]
    assert_partition(output, documents, manifest)


def test_corrupt_source_and_output_fail_closed(tmp_path):
    documents = [[1, 2], [3, 4], [5, 6], [7, 8]]
    source = make_source(tmp_path / "source", documents)
    idx = source / f"{SOURCE_PREFIX}.idx"
    content = idx.read_bytes()
    idx.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        read_index(idx)
    idx.write_bytes(content)
    with pytest.raises(ValueError, match="vocabulary"):
        prepare(source, tmp_path / "bad_vocab", vocabulary_size=4, block_tokens=1)
    prepare(source, tmp_path / "complete")
    path = tmp_path / "complete" / "A.bin"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file size"):
        prepare(source, tmp_path / "complete")
