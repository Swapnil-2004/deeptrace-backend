"""
DeepTrace - tools/baseline.py
Runs the CURRENT pipeline on every test file and saves all scores + peak RAM
into a JSON file. Used as the "truth" to compare later changes against.

Run from the repo root (D:\\deeptrace\\deeptrace-lean) with the venv active:
    python tools\\baseline.py --tag original
Options:
    --testset D:\\deeptrace\\testset     folder that holds images/, videos/, images_big/, videos_big/
    --only images|images_big|videos|videos_big   (run just one folder)
    --force                              re-run files that are already in the results file
"""
import argparse, hashlib, json, os, sys, threading, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
import numpy as np
import psutil

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


class PeakRSS:
    """Samples this process' RAM every 20 ms and remembers the highest value."""
    def __init__(self):
        self.proc = psutil.Process(os.getpid()); self.peak = 0; self._run = False
    def _loop(self):
        while self._run:
            self.peak = max(self.peak, self.proc.memory_info().rss); time.sleep(0.02)
    def __enter__(self):
        self.peak = self.proc.memory_info().rss; self._run = True
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self._run = False; self.t.join(); self.peak = max(self.peak, self.proc.memory_info().rss)


def jsonable(o):
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def label_of(name):
    n = name.lower()
    return "real" if "real" in n else ("fake" if "fake" in n else "unknown")


def image_info(path):
    from PIL import Image
    with Image.open(path) as im:
        return {"width": im.size[0], "height": im.size[1]}


def video_info(path):
    import cv2
    c = cv2.VideoCapture(str(path))
    d = {"width": int(c.get(3)), "height": int(c.get(4)), "fps": round(c.get(5), 2), "frames": int(c.get(7))}
    c.release(); return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="original")
    ap.add_argument("--testset", default=r"D:\deeptrace\testset")
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import torch, cv2
    torch.set_num_threads(1); cv2.setNumThreads(1)          # same as app.py startup
    from detector import pipeline

    root = Path(a.testset); outdir = root / "results"; outdir.mkdir(exist_ok=True)
    out = outdir / f"results_{a.tag}.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    print(f"[baseline] tag={a.tag}  results file: {out}  (already have {len(results)} entries)")

    folders = [a.only] if a.only else ["images", "images_big", "videos", "videos_big"]
    todo = []
    for f in folders:
        d = root / f
        if d.is_dir():
            for p in sorted(d.iterdir()):
                ext = p.suffix.lower()
                if ext in IMG_EXT or ext in VID_EXT:
                    todo.append((f, p))
    print(f"[baseline] {len(todo)} files to consider")

    start_rss = psutil.Process().memory_info().rss / 1e6
    for i, (folder, p) in enumerate(todo, 1):
        key = f"{folder}/{p.name}"
        if key in results and not a.force and "error" not in results[key]:
            print(f"[{i}/{len(todo)}] skip {key} (done)"); continue
        is_video = p.suffix.lower() in VID_EXT
        entry = {"label": label_of(p.name), "kind": "video" if is_video else "image",
                 "file_mb": round(p.stat().st_size / 1e6, 2)}
        try:
            entry.update(video_info(p) if is_video else image_info(p))
            t0 = time.time()
            with PeakRSS() as pk:
                if is_video:
                    res = pipeline.analyse_video_file(str(p), p.name)
                else:
                    res = pipeline.analyse_image_bytes(p.read_bytes(), p.name)
            hm = res.pop("heatmap_base64", "") or ""
            res["heatmap_sha1"] = hashlib.sha1(hm.encode()).hexdigest() if hm else ""
            res["heatmap_len"] = len(hm)
            entry["seconds"] = round(time.time() - t0, 2)
            entry["peak_rss_mb"] = round(pk.peak / 1e6)
            entry["result"] = res
            print(f"[{i}/{len(todo)}] {key}  label={entry['label']}  pred={res.get('prediction')}  "
                  f"score={res.get('final_score')}  {entry['seconds']}s  peakRAM={entry['peak_rss_mb']}MB")
        except Exception as e:
            traceback.print_exc()
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {key}  ERROR: {entry['error']}")
        results[key] = entry
        out.write_text(json.dumps(results, indent=1, default=jsonable))   # save after every file

    ok = [v for v in results.values() if "error" not in v]
    print("\n===== SUMMARY =====")
    print(f"files done: {len(ok)}   errors: {len(results) - len(ok)}   start RAM: {start_rss:.0f} MB")
    for kind in ("image", "video"):
        rows = [v for v in ok if v["kind"] == kind]
        if rows:
            hit = sum(1 for v in rows if v["label"] != "unknown" and
                      str(v["result"].get("prediction", "")).lower().startswith(v["label"][:4]))
            print(f"{kind}s: {len(rows)}  matched-label (informational): {hit}  "
                  f"max peak RAM: {max(v['peak_rss_mb'] for v in rows)} MB  "
                  f"slowest: {max(v['seconds'] for v in rows)} s")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
