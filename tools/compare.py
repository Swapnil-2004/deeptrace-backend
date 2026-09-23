"""
DeepTrace - tools/compare.py
Compares two results files made by baseline.py (old code vs new code).
Prints PASS/FAIL per file: same verdict + all numbers within tolerance.

    python tools\\compare.py original lean1
    python tools\\compare.py original lean_fp16 --tol 0.005     (approximate changes)
Default tolerance is 1e-4 (for memory-only changes that should not change numbers).
"""
import argparse, json, sys
from pathlib import Path


def flat(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{prefix}{k}.")
    elif isinstance(o, list) and o and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
        for i, x in enumerate(o): yield f"{prefix}[{i}]", x
    elif isinstance(o, list):
        yield prefix.rstrip("."), json.dumps(o, sort_keys=True)
    else:
        yield prefix.rstrip("."), o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--testset", default=r"D:\deeptrace\testset")
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    d = Path(a.testset) / "results"
    old = json.loads((d / f"results_{a.old}.json").read_text())
    new = json.loads((d / f"results_{a.new}.json").read_text())

    fails = 0; compared = 0; worst = 0.0
    mem_old = mem_new = 0
    for key, o in old.items():
        n = new.get(key)
        if n is None: print(f"MISSING in new: {key}"); fails += 1; continue
        if "error" in o or "error" in n:
            print(f"ERROR   {key}: old={o.get('error')} new={n.get('error')}"); fails += (o.get('error') != n.get('error')); continue
        compared += 1
        problems = []
        fo, fn = dict(flat(o["result"])), dict(flat(n["result"]))
        for path, vo in fo.items():
            vn = fn.get(path, "<missing>")
            if isinstance(vo, (int, float)) and not isinstance(vo, bool) and isinstance(vn, (int, float)):
                diff = abs(vo - vn); worst = max(worst, diff)
                if diff > a.tol: problems.append(f"{path}: {vo} -> {vn} (diff {diff:.2e})")
            elif vo != vn and not path.startswith("heatmap_"):
                problems.append(f"{path}: {vo!r} -> {vn!r}")
        if o["result"].get("prediction") != n["result"].get("prediction"):
            problems.insert(0, f"VERDICT CHANGED: {o['result'].get('prediction')} -> {n['result'].get('prediction')}")
        mem_old = max(mem_old, o["peak_rss_mb"]); mem_new = max(mem_new, n["peak_rss_mb"])
        heat = "same heatmap" if o["result"].get("heatmap_sha1") == n["result"].get("heatmap_sha1") else "heatmap differs"
        if problems:
            fails += 1; print(f"FAIL {key}  [{heat}]"); [print("     ", p) for p in problems[:8]]
        else:
            print(f"PASS {key}  [{heat}]  RAM {o['peak_rss_mb']} -> {n['peak_rss_mb']} MB")
    print(f"\ncompared {compared} files, tolerance {a.tol}, largest number difference {worst:.2e}")
    print(f"highest peak RAM: old {mem_old} MB -> new {mem_new} MB")
    print("RESULT:", "ALL PASS" if fails == 0 else f"{fails} FILE(S) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
