#!/usr/bin/env python3
"""Multi-chunk parallel downloader (Range requests). Defaults to 16 parallel chunks per file."""

import sys, os, requests, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_chunk(url, start, end, dest, chunk_idx):
    """Download bytes [start, end] (inclusive) to dest at offset start."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest, "r+b") as f:
                f.seek(start)
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
            return chunk_idx, end - start + 1
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  chunk {chunk_idx} retry {attempt+1}: {e}", flush=True)
    return chunk_idx, 0


def get_size(url):
    r = requests.head(url, allow_redirects=True, timeout=30)
    if 'Content-Length' in r.headers:
        return int(r.headers['Content-Length'])
    # Fallback: GET with Range 0-0
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
    return int(r.headers.get('Content-Range', '/0').split('/')[-1])


def parallel_download(url, dest, n_chunks=16, resume=True):
    total = get_size(url)
    print(f"Total: {total/1024/1024:.1f} MB")

    # Pre-allocate file or resume
    existing = os.path.exists(dest) and os.path.getsize(dest) == total
    if existing:
        print("File already complete")
        return

    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        # Truncate file to exact size for offset writing
        with open(dest, "wb") as f:
            f.seek(total - 1)
            f.write(b'\0')

    chunk_size = total // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = total - 1 if i == n_chunks - 1 else (i + 1) * chunk_size - 1
        chunks.append((start, end, i))

    downloaded = [0] * n_chunks
    completed = [0]

    def worker(args):
        start, end, idx = args
        return download_chunk(url, start, end, dest, idx)

    print(f"Starting {n_chunks} parallel chunks...")
    with ThreadPoolExecutor(max_workers=n_chunks) as ex:
        futures = {ex.submit(worker, c): c for c in chunks}
        for f in as_completed(futures):
            try:
                idx, sz = f.result()
                downloaded[idx] = sz
                completed[0] += 1
                total_so_far = sum(downloaded) / 1024 / 1024
                print(f"  chunk {idx} done ({completed[0]}/{n_chunks}), total {total_so_far:.0f} MB", flush=True)
            except Exception as e:
                print(f"FAIL: {e}", flush=True)
                raise
    print(f"Done: {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: multi_chunk_download.py URL DEST [N_CHUNKS]")
        sys.exit(1)
    url = sys.argv[1]
    dest = sys.argv[2]
    nchunks = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    parallel_download(url, dest, nchunks)
