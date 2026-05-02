#!/usr/bin/env python3
"""
Incremental asset synchronizer.

Usage: sync_assets.py <engine_src> <app_src> <build_dir>

Copies files from <engine_src> into <build_dir>, then overlays files from
<app_src>. Only files that are new or have different mtime/size are copied.
Files present in <build_dir> but not in either source are removed.
"""
import sys
import os
import shutil
from pathlib import Path


def build_file_index(root: Path):
    files = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            try:
                st = p.stat()
                files[rel] = (st.st_mtime, st.st_size)
            except OSError:
                # Skip unreadable files
                continue
    return files


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src: Path, dst: Path, src_stat):
    if not dst.exists():
        ensure_parent(dst)
        shutil.copy2(src, dst)
        return True
    try:
        dst_stat = dst.stat()
    except OSError:
        ensure_parent(dst)
        shutil.copy2(src, dst)
        return True
    # Compare size and mtime (with small tolerance)
    if dst_stat.st_size != src_stat[1] or abs(dst_stat.st_mtime - src_stat[0]) > 1e-6:
        shutil.copy2(src, dst)
        return True
    return False


def main(argv):
    if len(argv) < 4:
        print("Usage: sync_assets.py <engine_src> <app_src> <build_dir>", file=sys.stderr)
        return 2

    engine_src = Path(argv[1])
    app_src = Path(argv[2])
    build_dir = Path(argv[3])

    if not engine_src.exists() and not app_src.exists():
        print("Neither engine nor app source directories exist", file=sys.stderr)
        return 3

    engine_files = build_file_index(engine_src) if engine_src.exists() else {}
    app_files = build_file_index(app_src) if app_src.exists() else {}

    # Ensure build dir exists
    build_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    # Copy engine files first
    for rel, stat in engine_files.items():
        src = engine_src / rel
        dst = build_dir / rel
        if copy_if_needed(src, dst, stat):
            copied += 1

    # Overlay app files
    for rel, stat in app_files.items():
        src = app_src / rel
        dst = build_dir / rel
        if copy_if_needed(src, dst, stat):
            copied += 1

    # Remove files from build that are not in engine_files or app_files
    valid = set(engine_files.keys()) | set(app_files.keys())
    removed = 0
    for p in build_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(build_dir).as_posix()
            if rel not in valid:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass

    # Optionally remove empty directories
    for d in sorted(build_dir.rglob("*"), key=lambda x: -len(str(x))):
        if d.is_dir():
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                except OSError:
                    pass

    print(f"sync_assets: copied={copied} removed={removed}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
