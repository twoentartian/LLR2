"""LMDB utilities for storing/loading model checkpoints.

Ported from DFL_torch/py_src/lmdb_pack.py.
"""

import os
import re


def generate_lmdb_index_from_node_name_and_tick(node_name: int, tick: int) -> str:
    return f"{node_name}/{tick}.model.pt"


def get_node_name_and_tick_from_lmdb_index(lmdb_index) -> tuple:
    match = re.match(rb"(\d+)/(\d+)\.model\.pt", lmdb_index)
    node_name = int(match.group(1))
    tick = int(match.group(2))
    return node_name, tick


def store_folder_in_lmdb(root_folder: str, lmdb_path: str, map_size: int = 1099511627776):
    import lmdb
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_folder)
                with open(file_path, 'rb') as f:
                    txn.put(relative_path.encode(), f.read())
    env.close()


def load_folder_from_lmdb(lmdb_path: str, output_folder: str):
    import lmdb
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        for key, value in txn.cursor():
            relative_path = key.decode()
            file_path = os.path.join(output_folder, relative_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(value)
    env.close()
