#!/usr/bin/env python3
"""Patch DiscoRL meta_nets for JAX tracer compatibility."""
from __future__ import annotations

from pathlib import Path


def patch_file(path: Path) -> bool:
    text = path.read_text()
    updated = text

    updated = updated.replace(
        "    assert isinstance(logits, chex.Array)\n",
        "    if not hasattr(logits, 'shape'):\n"
        "      raise TypeError('Expected array-like logits with a shape attribute.')\n",
    )
    updated = updated.replace(
        "  if isinstance(y, chex.Array) and y.shape:  # not scalar\n",
        "  if hasattr(y, 'shape') and y.shape:  # not scalar\n",
    )

    if updated == text:
        print(f"No changes needed in {path}")
        return False

    path.write_text(updated)
    print(f"Patched {path}")
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "external/disco_rl/disco_rl/networks/meta_nets.py"
    if not target.exists():
        raise SystemExit(f"Target file not found: {target}")
    patch_file(target)


if __name__ == "__main__":
    main()
