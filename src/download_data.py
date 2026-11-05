"""
Utility script to download the 'nonameee12233/job-resume-matching' dataset
from Hugging Face into the local data/ folder.
"""

import os
import sys
import time
import urllib.request

DATA_URLS = {
    "job.csv": "https://huggingface.co/datasets/nonameee12233/job-resume-matching/resolve/main/job.csv",
    "matches.csv": "https://huggingface.co/datasets/nonameee12233/job-resume-matching/resolve/main/matches.csv",
    "cv.csv": "https://huggingface.co/datasets/nonameee12233/job-resume-matching/resolve/main/cv.csv"
}

def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[INFO] {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path) / (1024*1024):.2f} MB). Skipping download.")
        return

    print(f"[INFO] Downloading {os.path.basename(dest_path)} from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB chunk
        start_time = time.time()
        last_print = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 2 or downloaded == total_size:
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                speed = (downloaded / (1024 * 1024)) / max(now - start_time, 0.001)
                print(f"  -> {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                last_print = now

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Downloaded {os.path.basename(dest_path)} in {elapsed:.1f}s.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Target data directory: {data_dir}")
    for filename, url in DATA_URLS.items():
        dest_path = os.path.join(data_dir, filename)
        download_file(url, dest_path)

    print("\n[ALL COMPLETE] All dataset files are in place.")

if __name__ == "__main__":
    main()

