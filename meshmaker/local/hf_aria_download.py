#!/usr/bin/env python3
"""Mirror a Hugging Face repo to a local directory using aria2c.

Built for slow/flaky networks: aria2c opens N parallel connections per file
(--split), auto-resumes (.aria2 control files), and retries transient errors.
Only stdlib + aria2c required.

Usage:
    python3 hf_aria_download.py <repo_id> <dest_dir> [--conn N] [--jobs N]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


def list_files(repo: str) -> list[dict]:
    req = urllib.request.Request(API.format(repo=repo), headers={"User-Agent": "hf-aria-dl"})
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  list attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(min(60, 2 ** attempt))
    raise SystemExit(f"could not list {repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--conn", type=int, default=16, help="connections per file")
    ap.add_argument("--jobs", type=int, default=4, help="parallel files")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().absolute()
    dest.mkdir(parents=True, exist_ok=True)

    files = list_files(args.repo)
    entries = []
    total = 0
    for f in files:
        if f["type"] != "file":
            continue
        path = f["path"]
        if path.endswith((".gitattributes",)):
            continue
        total += f.get("size", 0)
        out = dest / path
        if out.exists() and out.stat().st_size == f.get("size"):
            continue  # already complete
        out.parent.mkdir(parents=True, exist_ok=True)
        entries.append((RESOLVE.format(repo=args.repo, path=path), str(out)))

    print(f"{args.repo}: {len(entries)} files to fetch, {total / 1e9:.1f} GB repo total")
    if not entries:
        print("everything already downloaded")
        return

    # aria2c input file: url + local path
    listfile = dest / ".aria_input.txt"
    listfile.write_text("".join(f"{u}\n  out={Path(p).relative_to(dest)}\n" for u, p in entries))

    cmd = [
        "aria2c", "-i", str(listfile), "-d", str(dest),
        "-x", str(args.conn), "-s", str(args.conn), "-j", str(args.jobs),
        "-c",  # continue partial downloads
        "--retry-wait=5", "--max-tries=0", "--timeout=60",
        "--connect-timeout=30",
        "--file-allocation=none",
        "--summary-interval=30",
        "--console-log-level=warn",
    ]
    while True:
        rc = subprocess.call(cmd)
        if rc == 0:
            print("download complete")
            listfile.unlink(missing_ok=True)
            return
        print(f"aria2c exited {rc}; resuming in 15s...", file=sys.stderr)
        time.sleep(15)


if __name__ == "__main__":
    main()
