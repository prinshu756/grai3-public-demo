"""Rebuild the test submission from the published per-view demo results.

The raw views live in the public repository (data/grai3_d8_views.json, 279 views = 8 per photo),
so nothing here needs a browser or any local model: it is pure post-processing of published numbers.

    python make_sub.py                        # verify only (must reproduce the submitted 10.03072 file)
    python make_sub.py 0.03 -0.02 out.csv     # apply b *= exp(0.03), c *= exp(-0.02)
    python make_sub.py --raw=path.json 0 0 out.csv   # use a different raw-view file
"""
import json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "grai3-public-demo", "data", "grai3_d8_views.json")
V7 = os.path.join(HERE, "..", "grai3-public-demo", "submission.csv")
DATA = r"D:\SOIL_GRAIN_SIZE_FROM_PHOTOS\soil-grain-size-from-photos"
SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]


def bc_from_diameters(fr):
    pts = [(0.1, fr.get("d10")), (0.3, fr.get("d30")), (0.6, fr.get("d60"))]
    pts = [(p, v) for p, v in pts if v and v > 0]
    if len(pts) < 2:
        return None
    k = np.array([-np.log(1.0 - p) for p, _ in pts])
    A = np.column_stack([np.ones(len(pts)), np.log(k)])
    sol, *_ = np.linalg.lstsq(A, np.log([v for _, v in pts]), rcond=None)
    lnb, inv_c = sol
    if not (0.02 < inv_c < 25.0):
        return None
    b, c = float(np.exp(lnb)), float(1.0 / inv_c)
    if not (1e-3 < b < 1e3):
        return None
    return b, c


def norm_id(s):
    s = (s.replace("\u00dc", "Ue").replace("\u00fc", "ue").replace("\u00c4", "Ae")
          .replace("\u00e4", "ae").replace("\u00d6", "Oe").replace("\u00f6", "oe")
          .replace("\u00df", "ss").replace(",", "_"))
    return re.sub(r"\s+", " ", s).strip()


def sample_id(path):
    b = os.path.basename(path)
    b = re.sub(r"\.(jpg|jpeg)$", "", b, flags=re.I)
    b = re.sub(r"\s*\(\d+\)\s*$", "", b)
    i = b.find("HPC")
    return norm_id(b[i:] if i >= 0 else b)


def curves(raw, shift=(0.0, 0.0), tset=range(8), how="mean"):
    per_photo = {}
    for key, fr in raw.items():
        name, t, _ = key.rsplit("|", 2)
        if int(t) not in tset:
            continue
        bc = bc_from_diameters(fr)
        if bc is None:
            continue
        b, c = bc
        cv = np.clip((1.0 - np.exp(-((SUPPORT / (b * np.exp(shift[0]))) ** (c * np.exp(shift[1]))))) * 100.0, 0, 100)
        per_photo.setdefault(name, []).append(cv)
    photo = {k: (np.median(np.vstack(v), axis=0) if how == "median" else np.mean(np.vstack(v), axis=0))
             for k, v in per_photo.items()}
    per_sample = {}
    for name, cv in photo.items():
        per_sample.setdefault(sample_id(name), []).append(cv)
    out = {k: (np.median(np.vstack(v), axis=0) if how == "median" else np.mean(np.vstack(v), axis=0))
           for k, v in per_sample.items()}
    return out, len(photo)


def frame(tta):
    sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    rows = [tta[s] for s in sub["sample_id"]]
    df = pd.DataFrame(np.vstack(rows), columns=COLS)
    df.insert(0, "sample_id", sub["sample_id"].values)
    df[COLS] = np.maximum.accumulate(np.clip(df[COLS].values, 0, 100), axis=1)
    df["200"] = 100.0
    return df


def main():
    raw_path = RAW
    nviews = None
    args = []
    for a in sys.argv[1:]:
        if a.startswith("--raw="):
            raw_path = a.split("=", 1)[1]
        elif a.startswith("--views="):
            nviews = int(a.split("=", 1)[1])
        else:
            args.append(a)
    raw = json.load(open(raw_path))
    if nviews is None:
        nviews = 1 + max(int(k.rsplit("|", 2)[1]) for k in raw)
    print("raw views:", len(raw), "from", raw_path, "| transforms used:", nviews)
    shift = (float(args[0]), float(args[1])) if len(args) > 1 else (0.0, 0.0)
    out_path = args[2] if len(args) > 2 else None
    how = args[3] if len(args) > 3 else "mean"

    tta, nphoto = curves(raw, shift, tset=range(nviews), how=how)
    df = frame(tta)
    old = pd.read_csv(V7)
    d = np.abs(df[COLS].values - old[COLS].values)
    print(f"photos used: {nphoto}   shift: {shift}   aggregation: {how}")
    print("vs stored v7 submission: max %.3e  mean %.3e" % (d.max(), d.mean()))
    assert (np.diff(df[COLS].values, axis=1) >= -1e-9).all()
    assert np.allclose(df["200"].values, 100.0)
    if out_path:
        df.to_csv(out_path, index=False)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
