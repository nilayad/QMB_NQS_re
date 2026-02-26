"""
generate_repo_dump.py
---------------------
Collects the raw content of every text-based file in this repository and
writes them all into a single file: repo_contents.txt

Usage:
    python generate_repo_dump.py
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(REPO_ROOT, "repo_contents.txt")

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf",
    ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".dll", ".exe",
}
SKIP_DIRS = {
    ".git", "__pycache__", ".tox", "node_modules",
    ".eggs", "dist", "build", ".venv", "venv",
}
SKIP_FILES = {"repo_contents.txt"}


def collect_files(root):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        rel_dir = os.path.relpath(dirpath, root)
        for fname in sorted(filenames):
            if fname in SKIP_FILES:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SKIP_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.join(rel_dir, fname) if rel_dir != "." else fname
            entries.append((rel_path, full_path))
    return entries


def main():
    files = collect_files(REPO_ROOT)
    separator = "=" * 72

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("REPOSITORY CONTENTS DUMP\n")
        out.write(f"Generated from: {REPO_ROOT}\n")
        out.write(f"Total files: {len(files)}\n")
        out.write(separator + "\n\n")

        for rel_path, full_path in files:
            out.write(f"{separator}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{separator}\n")
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as exc:
                out.write(f"[ERROR reading file: {exc}]\n")
            out.write("\n\n")

    print(f"Done. Output written to: {OUTPUT_FILE}")
    print(f"Included {len(files)} files.")


if __name__ == "__main__":
    main()
