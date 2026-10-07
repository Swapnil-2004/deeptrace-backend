"""
DeepTrace - tools/memserve_api_lean.py
Memory per SERVICE: real vit_service.py + real app.py (VIT_SERVICE_URL set),
each in its own process. Sends a 5.8 MP photo (no face = worst case for BTD)
and a 720p video (job route). Run from the repo root with the venv-lean python:
    python tools\memserve_api_lean.py
"""
import os, sys, time, subprocess, threading, json, urllib.request
import psutil
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = r"D:\deeptrace\testset"
MB = lambda b: round(b / 1e6)
VPORT, APORT = 8101, 8102


def start(app, port, extra):
    env = dict(os.environ)
    env["DEEPTRACE_BACKEND"] = "onnx"
    env.pop("VIT_SERVICE_URL", None)
    env.update(extra)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--port", str(port)],
        cwd=ROOT, env=env, stdout=open(os.path.join(TEST, "memserve_%d.log" % port), "w"), stderr=subprocess.STDOUT)


class Tree:
    def __init__(self, p):
        self.root = psutil.Process(p.pid)
        self.peak = 0
        self.run = True
        threading.Thread(target=self.loop, daemon=True).start()

    def rss(self):
        total = 0
        try:
            for q in [self.root] + self.root.children(recursive=True):
                try:
                    total += q.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def loop(self):
        while self.run:
            self.peak = max(self.peak, self.rss())
            time.sleep(0.01)

    def reset(self):
        self.peak = self.rss()

    def stop(self):
        self.run = False
        try:
            for q in self.root.children(recursive=True):
                q.terminate()
            self.root.terminate()
        except Exception:
            pass


def wait(port):
    for _ in range(240):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def curl_form(path, url):
    r = subprocess.run(["curl.exe", "-s", "-w", "\nHTTPCODE %{http_code}",
                        "-F", "file=@" + path, url], capture_output=True, text=True)
    body, _, code = r.stdout.rpartition("\nHTTPCODE ")
    return code.strip(), body


def report(name, secs, ta, tv):
    print("%-14s %6.1fs | API peak %4d MB (after %4d) | ViT peak %4d MB (after %4d)"
          % (name, secs, MB(ta.peak), MB(ta.rss()), MB(tv.peak), MB(tv.rss())))


img = os.path.join(TEST, "mem_5_8mp.jpg")
if not os.path.exists(img):
    Image.open(os.path.join(TEST, "images_big", "real_big_01_12mp.jpg")).convert("RGB") \
        .resize((2900, 2000), Image.LANCZOS).save(img, quality=90)
vid = os.path.join(TEST, "videos", "fake_v02.mp4")

pv = start("vit_service:app", VPORT, {})
pa = start("app:app", APORT, {"VIT_SERVICE_URL": "http://127.0.0.1:%d" % VPORT})
tv, ta = Tree(pv), Tree(pa)
try:
    if not (wait(VPORT) and wait(APORT)):
        print("a service did not start")
        sys.exit(1)
    time.sleep(1)
    print("idle: API %d MB | ViT %d MB" % (MB(ta.rss()), MB(tv.rss())))
    base = "http://127.0.0.1:%d" % APORT

    tv.reset(); ta.reset(); t0 = time.time()
    code, body = curl_form(img, base + "/analyse/image")
    secs = time.time() - t0
    try:
        j = json.loads(body)
        info = "%s %s %s" % (j.get("prediction"), j.get("final_score"), j.get("detail"))
    except Exception:
        info = body[:120]
    print("IMAGE 5.8 MP: HTTP %s | %s" % (code, info))
    report("image 5.8 MP", secs, ta, tv)

    tv.reset(); ta.reset(); t0 = time.time()
    code, body = curl_form(vid, base + "/analyse/video/start")
    print("VIDEO start: HTTP %s | %s" % (code, body[:100]))
    job = json.loads(body)["job_id"]
    while True:
        time.sleep(3)
        s = json.loads(urllib.request.urlopen(base + "/analyse/video/status/" + job, timeout=30).read())
        if s["status"] != "running":
            break
    secs = time.time() - t0
    res = s.get("result", {})
    print("VIDEO 720p: %s | %s %s | %s" % (s["status"], res.get("prediction"), res.get("final_score"), s.get("detail")))
    report("video 720p", secs, ta, tv)
finally:
    ta.stop()
    tv.stop()

