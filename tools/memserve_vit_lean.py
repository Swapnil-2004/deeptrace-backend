"""
DeepTrace - tools/memserve_vit_lean.py
Memory of the ViT service ALONE (real vit_service.py, uvicorn + FastAPI included).
Run from the repo root with the venv-lean python:
    python tools\memserve_vit_lean.py
"""
import os, sys, time, subprocess, threading, urllib.request
import numpy as np
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8101
MB = lambda b: round(b / 1e6)
env = dict(os.environ)
env["DEEPTRACE_BACKEND"] = "onnx"
env.pop("VIT_SERVICE_URL", None)
p = subprocess.Popen([sys.executable, "-m", "uvicorn", "vit_service:app", "--port", str(PORT)],
                     cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
root = psutil.Process(p.pid)
state = {"peak": 0, "run": True}


def tree_rss():
    total = 0
    try:
        for q in [root] + root.children(recursive=True):
            try:
                total += q.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass
    return total


def sampler():
    while state["run"]:
        state["peak"] = max(state["peak"], tree_rss())
        time.sleep(0.01)


threading.Thread(target=sampler, daemon=True).start()
base = "http://127.0.0.1:%d" % PORT
ok = False
for _ in range(120):
    try:
        urllib.request.urlopen(base + "/health", timeout=2).read()
        ok = True
        break
    except Exception:
        time.sleep(0.5)
if not ok:
    print("service did not start")
    p.kill()
    sys.exit(1)
time.sleep(1)
idle = tree_rss()
print("idle after start, model NOT loaded yet (MB):", MB(idle))
body = np.random.default_rng(0).integers(0, 256, 224 * 224 * 3, dtype=np.uint8).tobytes()


def call():
    req = urllib.request.Request(base + "/predict", data=body, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    return urllib.request.urlopen(req, timeout=300).read()


state["peak"] = idle
t0 = time.time()
r = call()
dt = time.time() - t0
print("1st request (loads the model): %.1fs | peak %d MB | after %d MB | reply %s"
      % (dt, MB(state["peak"]), MB(tree_rss()), r.decode()))
state["peak"] = tree_rss()
t0 = time.time()
for _ in range(30):
    call()
print("30 more requests: %.1fs total | peak %d MB | after %d MB"
      % (time.time() - t0, MB(state["peak"]), MB(tree_rss())))
state["run"] = False
p.terminate()
