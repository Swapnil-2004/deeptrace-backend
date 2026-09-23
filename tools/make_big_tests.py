"""
DeepTrace - tools/make_big_tests.py
Your test files are small, so they cannot show memory problems. This makes a few BIG
copies (12MP and 24MP images, 1080p and 4K videos) by enlarging your existing files.
These big files are only for memory + old-vs-new comparison, NOT for judging accuracy.
Real phone photos are even better: put them in images_big as real_big_XX.jpg / fake_big_XX.jpg.

    python tools\\make_big_tests.py
"""
import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default=r"D:\deeptrace\testset")
    a = ap.parse_args(); root = Path(a.testset)
    (root / "images_big").mkdir(exist_ok=True); (root / "videos_big").mkdir(exist_ok=True)

    # ---- big images: 2 real + 2 fake, each at 12MP (4000x3000) and 24MP (6000x4000)
    picks = [("real", sorted((root / "images").glob("real_*"))[:2]),
             ("fake", sorted((root / "images").glob("fake_*"))[:2])]
    for label, files in picks:
        for f in files:
            img = cv2.imread(str(f))
            if img is None: continue
            for name, (w, h) in (("12mp", (4000, 3000)), ("24mp", (6000, 4000))):
                big = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
                out = root / "images_big" / f"{label}_big_{f.stem.split('_')[-1]}_{name}.jpg"
                cv2.imwrite(str(out), big, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"made {out.name}  {out.stat().st_size/1e6:.1f} MB")

    # ---- big videos: 1 real + 1 fake, enlarged to 1080p and 4K
    for label in ("real", "fake"):
        src = sorted((root / "videos").glob(f"{label}_v*"))[:1]
        for f in src:
            for name, (w, h) in (("1080p", (1920, 1080)), ("4k", (3840, 2160))):
                cap = cv2.VideoCapture(str(f)); fps = cap.get(5) or 25.0
                out = root / "videos_big" / f"{label}_big_v_{name}.mp4"
                vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                n = 0
                while n < 150:                       # at most 150 frames (~6 s)
                    ok, fr = cap.read()
                    if not ok: break
                    vw.write(cv2.resize(fr, (w, h), interpolation=cv2.INTER_CUBIC)); n += 1
                cap.release(); vw.release()
                mb = out.stat().st_size / 1e6
                print(f"made {out.name}  {n} frames  {mb:.1f} MB" + ("   <-- over the 50MB app limit!" if mb > 50 else ""))

if __name__ == "__main__":
    main()
