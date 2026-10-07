"""
DeepTrace - tools/memprobe_video_lean.py
Memory per pipeline step for ONE video (lean ONNX path, no torch).
Runs the real analyse_video_file; splits RAM by the "[Pipeline] Step N" and
"Frame i/N" print lines. Changes no detection code.
Run from repo root with the venv-lean python:
    python tools\memprobe_video_lean.py D:\deeptrace\testset\videos_big\fake_big_v_1080p.mp4
"""
import os, sys, threading, time
os.environ["DEEPTRACE_BACKEND"] = "onnx"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil

proc = psutil.Process(os.getpid())
MB = lambda b: round(b / 1e6)
state = {"peak": 0, "run": True}
segs = []
cur = {"label": "start", "start": 0}


def sampler():
    while state["run"]:
        state["peak"] = max(state["peak"], proc.memory_info().rss)
        time.sleep(0.01)


def mark(label):
    now = proc.memory_info().rss
    segs.append((cur["label"], cur["start"], max(state["peak"], now)))
    cur["label"] = label
    cur["start"] = now
    state["peak"] = now


class Tee:
    def __init__(self, real):
        self.real = real

    def write(self, s):
        t = s.strip()
        if t.startswith("[Pipeline] Step") or (t.startswith("Frame ") and "/" in t):
            mark(t[:46])
        return self.real.write(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, n):
        return getattr(self.real, n)


video = sys.argv[1]
threading.Thread(target=sampler, daemon=True).start()
import detector.pipeline as pl
if '--no-vit' in sys.argv:
    import detector.vit_model as _vm
    _vm.predict = lambda img: 0.5
    print('NOTE: ViT replaced by 0.5 in this probe (API-side memory only)')
real_out = sys.stdout
imported = proc.memory_info().rss
print("RAM after importing detector.pipeline (MB):", MB(imported))
cur["start"] = imported
state["peak"] = imported
sys.stdout = Tee(real_out)
t0 = time.time()
result = pl.analyse_video_file(video, os.path.basename(video))
mark("end")
sys.stdout = real_out
state["run"] = False
secs = round(time.time() - t0, 1)

print("\n=== MEMORY PER STEP (MB): start -> highest ===")
frames = [s for s in segs if s[0].startswith("Frame ")]
for label, start, peak in segs:
    if label.startswith("Frame "):
        continue
    print(f"{label:<48}{MB(start):>6} -> {MB(peak):>6}")
if frames:
    worst = max(frames, key=lambda s: s[2])
    print(f"Frame loop: {len(frames)} frames, first starts {MB(frames[0][1])}, "
          f"highest {MB(worst[2])} ({worst[0]}), last frame highest {MB(frames[-1][2])}")
print("OVERALL PEAK (MB):", MB(max(s[2] for s in segs)))
print("RAM at the end (MB):", MB(proc.memory_info().rss))
print("seconds:", secs)
print("torch loaded:", "torch" in sys.modules, "| timm:", "timm" in sys.modules,
      "| transformers:", "transformers" in sys.modules)
print("verdict:", result.get("verdict"), "| final_score:", result.get("final_score"))

