"""Pre-compute molecular graphs and cache them for fast training.

This module provides:
- build_graph_database(): compute graphs from .xyz files and save as .pt
- CachedGraphDataset: load pre-computed graphs from .pt into memory

Typical workflow:
    1. Split .xyz files into train/val/test (notebook 1)
    2. Call build_graph_database() for each split (notebook 1, end)
    3. Use CachedGraphDataset in training (notebook 2)
"""

import hashlib
import random
import time
from glob import glob
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from torch_geometric.data import Data, Dataset

from gcnn.graphs.base import MolecularGraphs


def build_graph_database(
    xyz_dir: Path,
    output_path: Path,
    graphs: MolecularGraphs,
    n_max_entries: Optional[int] = None,
    seed: int = 42,
    config: Optional[dict] = None,
) -> int:
    """Pre-compute molecular graphs from .xyz files and save as a single .pt file.

    Args:
        xyz_dir: Directory containing .xyz molecule files.
        output_path: Path where the .pt file will be saved.
        graphs: A MolecularGraphs instance that converts files to Data objects.
        n_max_entries: If set, limit to this many molecules (randomly sampled).
        seed: Random seed for reproducible subsampling.
        config: Optional config dict to store as metadata for invalidation.

    Returns:
        Number of graphs successfully processed.
    """
    xyz_dir = Path(xyz_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob(str(xyz_dir / "*.xyz")))

    if not files:
        raise FileNotFoundError(f"No .xyz files found in {xyz_dir}")

    if n_max_entries and n_max_entries < len(files):
        random.seed(seed)
        files = random.sample(files, n_max_entries)

    data_list: List[Data] = []
    n_errors = 0
    t0 = time.time()

    for i, file in enumerate(files):
        try:
            graph = graphs.molecule2graph(file)
            data_list.append(graph)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  WARNING: skipping {file} — {e}")
            elif n_errors == 6:
                print("  (further warnings suppressed)")

        if (i + 1) % 500 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(files)}] {rate:.0f} graphs/s")

    # Save the graph list
    torch.save(data_list, output_path)

    # Save metadata alongside the .pt file
    metadata_path = output_path.with_suffix(".meta.yml")
    metadata = {
        "n_graphs": len(data_list),
        "n_errors": n_errors,
        "n_files_total": len(files),
        "source_dir": str(xyz_dir),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if config:
        metadata["config_hash"] = _config_hash(config)

    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    elapsed = time.time() - t0
    print(f"  Done: {len(data_list)} graphs saved to {output_path} ({elapsed:.1f}s)")

    return len(data_list)


class CachedGraphDataset(Dataset):
    """PyG Dataset that loads pre-computed graphs from a .pt file into memory.

    All graphs are held in RAM for instant access during training.
    """

    def __init__(
        self,
        pt_path: Path,
        transform=None,
        config: Optional[dict] = None,
    ) -> None:
        """Load graphs from a .pt file.

        Args:
            pt_path: Path to the .pt file created by build_graph_database().
            transform: Optional PyG transform to apply on each access.
            config: If provided, compare against stored metadata to warn
                    about stale caches.
        """
        super().__init__(transform=transform)
        pt_path = Path(pt_path)

        if not pt_path.exists():
            raise FileNotFoundError(
                f"Cached graph database not found: {pt_path}\n"
                f"Run build_graph_database() first."
            )

        # Check for stale cache
        if config:
            _check_metadata(pt_path, config)

        self._data_list: List[Data] = torch.load(pt_path, weights_only=False)

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]


def _config_hash(config: dict) -> str:
    """Compute a short hash of the relevant config keys for cache invalidation."""
    relevant_keys = [
        "EdgeFeatures", "AngleFeatures", "DihedralFeatures",
        "nodeFeatures", "nTotalNodeFeatures", "graphType",
        "nMaxNeighbours", "useCovalentRadii",
    ]
    subset = {k: config[k] for k in relevant_keys if k in config}
    raw = yaml.dump(subset, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _check_metadata(pt_path: Path, config: dict) -> None:
    """Warn if the cached .pt was built with a different config."""
    meta_path = pt_path.with_suffix(".meta.yml")
    if not meta_path.exists():
        return

    with open(meta_path) as f:
        metadata = yaml.safe_load(f)

    stored_hash = metadata.get("config_hash")
    if stored_hash and stored_hash != _config_hash(config):
        print(
            f"  ⚠ WARNING: cached graphs in {pt_path.name} were built with a "
            f"different config (hash mismatch). Consider rebuilding with "
            f"build_graph_database()."
        )
