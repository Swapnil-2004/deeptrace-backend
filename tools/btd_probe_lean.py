"""btd_probe_lean.py - peak RAM of ONE BTD signal on a full image (fresh process per signal).
Usage: python tools\btd_probe_lean.py <image_name_or_path> <signal letter C|D|E|F|G|H>
Read-only: imports detector.btd, changes nothing."""
import os, sys, gc, time, threading
os.environ["DEEPTRACE_BACKEND"] = "onnx"
sys.path.insert(0, os.getcwd())
import psutil
from PIL import Image

name, letter = sys.argv[1], sys.argv[2].upper()
path = name
if not os.path.isfile(path):
    for root, _, files in os.walk(r"D:\deeptrace\testset"):
        if name in files:
            path = os.path.join(root, name); break
img = Image.open(path).convert("RGB")
w, h = img.size

import detector.btd as btd
fn = {"C": btd.signal_c_edge_persistence, "D": btd.signal_d_ela,
      "E": btd.signal_e_noise, "F": btd.signal_f_chromatic_aberration,
      "G": btd.signal_g_colour_kurtosis, "H": btd.signal_h_high_frequency}[letter]

proc = psutil.Process()
gc.collect(); time.sleep(0.3)
base = proc.memory_info().rss
peak = [base]; stop = [False]
def sample():
    while not stop[0]:
        r = proc.memory_info().rss
        if r > peak[0]: peak[0] = r
        time.sleep(0.005)
t = threading.Thread(target=sample); t.start()
score, flag = fn(img)
stop[0] = True; t.join()
mb = 1e6
print("signal %s | %dx%d = %.1f MP | score %.6f | base %.0f MB | peak %.0f MB | signal cost +%.0f MB"
      % (letter, w, h, w*h/1e6, score, base/mb, peak[0]/mb, (peak[0]-base)/mb))
