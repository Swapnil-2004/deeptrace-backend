"""
DeepTrace - docker_fetch.py
Download ONE model file from the public Hugging Face repo and verify its SHA-256.
Used by Dockerfile.api / Dockerfile.vit. Exits with an error (build fails) on
any mismatch, so a wrong or corrupted file can never end up in an image.
Usage: python docker_fetch.py <file name> <sha256> [dest folder, default models]
"""
import hashlib
import os
import sys
import time
import urllib.request

BASE = "https://huggingface.co/Swapnil05092004/deeptrace-models/resolve/main/"
name, sha = sys.argv[1], sys.argv[2].lower()
dest_dir = sys.argv[3] if len(sys.argv) > 3 else "models"
os.makedirs(dest_dir, exist_ok=True)
dest = os.path.join(dest_dir, name)
last = None
for attempt in range(3):
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(BASE + name, timeout=300) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() == sha:
            print("[Build] fetched %s: %d bytes, sha256 OK" % (name, os.path.getsize(dest)))
            sys.exit(0)
        last = "HASH MISMATCH for %s: got %s" % (name, h.hexdigest())
    except SystemExit:
        raise
    except Exception as e:
        last = str(e)
    print("[Build] attempt %d failed: %s" % (attempt + 1, last))
    time.sleep(5)
if os.path.exists(dest):
    os.remove(dest)
print("[Build] FAILED to fetch %s: %s" % (name, last))
sys.exit(1)
