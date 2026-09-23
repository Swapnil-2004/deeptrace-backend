"""
DeepTrace - tools/memprobe.py
Shows WHERE the memory goes, step by step, for ONE file (image only for now).
It does not change any code: it wraps each pipeline step and records RAM
before / highest during / after.

Run from the repo root with the venv active:
    python tools\\memprobe.py D:\\deeptrace\\testset\\images\\real_04.jpeg
    python tools\\memprobe.py D:\\deeptrace\\testset\\images_big\\real_big_01_12mp.jpg
"""
import importlib, os, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
proc = psutil.Process(os.getpid())
MB = lambda b: round(b / 1e6)


class Peak:
    def __enter__(self):
        self.peak = proc.memory_info().rss; self.run = True
        def loop():
            while self.run:
                self.peak = max(self.peak, proc.memory_info().rss); time.sleep(0.01)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.run = False; self.t.join(); self.peak = max(self.peak, proc.memory_info().rss)


# ---------- Part 1: what does each library cost just to import? ----------
print("\n=== PART 1: RAM cost of importing each library (fresh process) ===")
print(f"{'library':<28}{'RAM after (MB)':>16}{'added (MB)':>14}")
last = proc.memory_info().rss
print(f"{'python only':<28}{MB(last):>16}{'':>14}")
for name in ["numpy", "cv2", "PIL", "scipy", "skimage", "torch", "timm", "transformers", "mediapipe", "onnxruntime"]:
    try:
        importlib.import_module(name)
    except Exception as e:
        print(f"{name:<28}{'not installed / failed':>30}"); continue
    now = proc.memory_info().rss
    print(f"{name:<28}{MB(now):>16}{MB(now - last):>14}"); last = now
import torch, cv2
torch.set_num_threads(1); cv2.setNumThreads(1)          # same as app.py
from detector import pipeline                            # imports the rest of your code
now = proc.memory_info().rss
print(f"{'detector.pipeline (yours)':<28}{MB(now):>16}{MB(now - last):>14}")

# ---------- Part 2: wrap each pipeline step ----------
rows = []
def wrap(module, fname, label):
    orig = getattr(module, fname)
    def inner(*a, **k):
        before = proc.memory_info().rss
        with Peak() as pk:
            out = orig(*a, **k)
        rows.append((label, MB(before), MB(pk.peak), MB(proc.memory_info().rss)))
        return out
    setattr(module, fname, inner)

from detector import face_utils, btd, effnet_model, gradcam, vit_model, metadata, ensemble
wrap(face_utils, "detect_and_crop",   "2 face detection (MediaPipe load+run)")
wrap(face_utils, "unload_landmarker", "  unload MediaPipe")
wrap(btd, "run_all_signals",          "3 BTD forensic signals")
wrap(effnet_model, "predict",         "4 EfficientNet (load+run)")
wrap(effnet_model, "unload_effnet",   "  unload EfficientNet")
wrap(gradcam, "generate_heatmap",     "5 Grad-CAM heatmap")
wrap(vit_model, "predict",            "6 ViT (load+run)")
wrap(vit_model, "unload_vit",         "  unload ViT")
wrap(metadata, "analyse_metadata",    "7 metadata")
wrap(ensemble, "compute_image_ensemble", "8 ensemble")

path = Path(sys.argv[1])
data = path.read_bytes()
import io
from PIL import Image
w, h = Image.open(io.BytesIO(data)).size
print(f"\n=== PART 2: step-by-step for {path.name}  ({w}x{h} = {w*h/1e6:.1f} MP) ===")
start = proc.memory_info().rss
with Peak() as total:
    pipeline.analyse_image_bytes(data, path.name)
print("\n" + "=" * 78)
print(f"{'step':<42}{'before':>9}{'highest':>9}{'after':>9}   (MB)")
for r in rows:
    print(f"{r[0]:<42}{r[1]:>9}{r[2]:>9}{r[3]:>9}")
print("=" * 78)
print(f"RAM before analysis: {MB(start)} MB    highest during analysis: {MB(total.peak)} MB    after: {MB(proc.memory_info().rss)} MB")
print("Target for Render free: 512 MB.")
