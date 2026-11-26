"""Score GRAI3 against the 24 labelled soils, then fit and cross-validate a correction.

Input : out_cal2/train_views.csv  (one row per photo: fitted Weibull b, c, and the ppm the demo used)
        Training_labels_updated.csv

Two correction families are compared, each validated two ways:
  * leave-one-soil-out  (does it generalise to an unseen soil?)
  * leave-one-camera-out (does it generalise to an unseen phone? -- the test photos are iPhones)
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = os.path.join(HERE, "out_cal2", "train_views.csv")
LABELS = r"D:\SOIL_GRAIN_SIZE_FROM_PHOTOS\soil-grain-size-from-photos\Training_labels_updated.csv"

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
W = np.diff(np.log10(SUPPORT))
assert abs(W.sum() - 5.0) < 1e-12


def curve_bc(b, c):
    return np.clip((1.0 - np.exp(-((SUPPORT / b) ** c))) * 100.0, 0.0, 100.0)


def score(pred, truth):
    d = np.abs(pred - truth)
    return float((d[:, 1:] * W).sum() / len(pred))


def load():
    df = pd.read_csv(VIEWS)
    df["photo"] = df["photo"].astype(str)
    if "camera" not in df:
        df["camera"] = np.where(df["photo"].str.startswith("Motorola_Edge_60_fusion"), "Edge60fusion",
                                np.where(df["photo"].str.startswith("Motorola_Edge"), "Edge", "Samsung"))
    lab = pd.read_csv(LABELS).set_index("sample_id")
    soils = sorted(df["soil"].unique())
    truth = np.maximum.accumulate(lab.loc[soils, COLS].values.astype(float), axis=1)
    return df, soils, truth


def aggregate(df, soils, shift=(0.0, 0.0), mode="curve"):
    lnb = np.log(df["b"].values) + shift[0]
    lnc = np.log(df["c"].values) + shift[1]
    cvs = np.vstack([curve_bc(np.exp(a), np.exp(c)) for a, c in zip(lnb, lnc)])
    out = []
    for s in soils:
        m = (df["soil"] == s).values
        if not m.any():
            out.append(None)
            continue
        if mode == "curve":
            out.append(cvs[m].mean(axis=0))
        else:
            out.append(curve_bc(np.exp(lnb[m].mean()), np.exp(lnc[m].mean())))
    return np.maximum.accumulate(np.vstack(out), axis=1)


def aggregate_how(df, soils, how="mean", trim=0.0):
    """Aggregate the per-photo curves of each soil with a chosen rule (metric is L1, so median is a candidate)."""
    cvs = np.vstack([curve_bc(b, c) for b, c in zip(df["b"].values, df["c"].values)])
    out = []
    for s in soils:
        m = (df["soil"] == s).values
        block = cvs[m]
        if how == "mean":
            out.append(block.mean(axis=0))
        elif how == "median":
            out.append(np.median(block, axis=0))
        elif how == "trimmed" and len(block) >= 3:
            k = int(np.floor(trim * len(block)))
            sb = np.sort(block, axis=0)
            out.append(sb[k:len(block) - k].mean(axis=0) if len(block) - 2 * k > 0 else block.mean(axis=0))
        else:
            out.append(block.mean(axis=0))
    return np.maximum.accumulate(np.vstack(out), axis=1)


def fit_shift(df, soils, truth):
    def obj(p):
        return score(aggregate(df, soils, shift=p), truth)
    best = None
    for x0 in ([0.0, 0.0], [0.15, 0.0], [-0.15, 0.0], [0.0, 0.15], [0.0, -0.15]):
        r = minimize(obj, x0=x0, method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def fit_scale(df, soils, truth):
    """One-parameter family: every diameter scaled by the same factor (b -> b s, c unchanged)."""
    def obj(x):
        return score(aggregate(df, soils, shift=(x, 0.0)), truth)
    r = minimize_scalar(obj, bounds=(-1.0, 1.0), method="bounded", options={"xatol": 1e-6})
    return float(r.x), float(r.fun)


def main():
    df, soils, truth = load()
    idx = {s: i for i, s in enumerate(soils)}
    print(f"photos: {len(df)}   soils: {len(soils)}   ppm values seen: {sorted(df['ppm_used'].dropna().unique())[:8]}")
    print("photos per camera:", df.groupby("camera").size().to_dict())
    print("photos per soil:", df.groupby("soil").size().to_dict())

    raw_curve = aggregate(df, soils, mode="curve")
    raw_par = aggregate(df, soils, mode="params")
    print(f"\nraw score, mean of curves : {score(raw_curve, truth):.5f}")
    print(f"raw score, mean of params : {score(raw_par, truth):.5f}")
    print(f"mean |dCDF|               : {np.abs(raw_curve - truth).mean():.3f} pp")

    d = raw_curve - truth
    print("\nper-bin (pred - true):")
    for j, s in enumerate(COLS):
        print(f"  {s:>7s}  signed {d[:, j].mean():+7.2f}   median {np.median(d[:, j]):+7.2f}   abs {np.abs(d[:, j]).mean():6.2f}")

    print("\nper soil:")
    for i, s in enumerate(soils):
        print(f"  {s}  score={float((np.abs(d[i, 1:]) * W).sum()):6.2f}  max={np.abs(d[i]).max():6.2f}")

    print("\nper camera (score on its soils, then mean signed error per sieve):")
    for cam in sorted(df["camera"].unique()):
        sc = sorted(df[df.camera == cam].soil.unique())
        a = aggregate(df[df.camera == cam], sc)
        t = truth[[idx[s] for s in sc]]
        e = (a - t).mean(axis=0)
        print(f"  {cam:12s} photos={int((df.camera == cam).sum()):3d} soils={len(sc):2d} "
              f"score={score(a, t):6.3f}")
        print("    " + " ".join(f"{v:+5.1f}" for v in e))

    print("\naggregation across the photos of a soil (the test set averages 3-5 photos, 8 views each):")
    for how, trim in (("mean", 0.0), ("median", 0.0), ("trimmed", 0.2), ("trimmed", 0.3)):
        print(f"  {how:8s} trim={trim:.1f}: {score(aggregate_how(df, soils, how, trim), truth):.5f}")

    # ---- diagnostic: how much does each individual photo want to be rescaled? ----
    # A photo taken at a different distance from the test protocol shows up as a per-photo scale.
    # If these are tight, a single global shift is meaningful; if they are wide, the acquisition
    # geometry varies and only the shape part of the correction can be trusted.
    def photo_scale(b, c, t):
        def obj(x):
            return float((np.abs(curve_bc(b * np.exp(x), c) - t)[1:] * W).sum())
        r = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-6})
        return float(r.x), float(r.fun)

    xs, fs = [], []
    for _, r in df.iterrows():
        x, f = photo_scale(r.b, r.c, truth[idx[r.soil]])
        xs.append(x); fs.append(f)
    df["scale_x"] = xs
    print("\nper-photo optimal diameter scale against its own soil's truth:")
    for cam in sorted(df["camera"].unique()):
        v = df.loc[df.camera == cam, "scale_x"].values
        print(f"  {cam:12s} n={len(v):3d}  median x{np.exp(np.median(v)):.3f}  "
              f"IQR x{np.exp(np.percentile(v, 25)):.3f}-{np.exp(np.percentile(v, 75)):.3f}  "
              f"std(ln) {np.std(v):.3f}")
    print(f"  {'all':12s} n={len(xs):3d}  median x{np.exp(np.median(xs)):.3f}  std(ln) {np.std(xs):.3f}")
    preds = np.vstack([curve_bc(r.b * np.exp(x), r.c) for (_, r), x in zip(df.iterrows(), xs)])
    agg = []
    for s in soils:
        m = (df["soil"] == s).values
        agg.append(preds[m].mean(axis=0))
    agg = np.maximum.accumulate(np.vstack(agg), axis=1)
    print(f"  score with a free per-photo scale (in-sample, 1 param/photo): {score(agg, truth):.5f}")
    df.to_csv(os.path.join(HERE, "out_cal2", "train_views_scaled.csv"), index=False)

    # ---- family A: global Weibull shift, validated by leave-one-soil-out ----
    p, f = fit_shift(df, soils, truth)
    print(f"\nA) global shift  dlnb={p[0]:+.4f}  dlnc={p[1]:+.4f}  (b x{np.exp(p[0]):.4f}, c x{np.exp(p[1]):.4f})")
    print(f"   in-sample {f:.5f}   raw {score(raw_curve, truth):.5f}")

    loo_raw, loo_cor = [], []
    for s in soils:
        keep = [t for t in soils if t != s]
        pp, _ = fit_shift(df[df.soil.isin(keep)], keep, truth[[idx[t] for t in keep]])
        loo_cor.append(aggregate(df[df.soil == s], [s], shift=pp)[0])
        loo_raw.append(raw_curve[idx[s]])
    print(f"   leave-one-soil-out: raw {score(np.vstack(loo_raw), truth):.5f} -> "
          f"corrected {score(np.vstack(loo_cor), truth):.5f}")

    # ---- family A validated by leaving out a whole camera ----
    print("\n   leave-one-camera-out:")
    for cam in ("Edge", "Edge60fusion", "Samsung"):
        tr = df[df.camera != cam]
        te = df[df.camera == cam]
        so_tr = sorted(tr.soil.unique())
        so_te = sorted(te.soil.unique())
        pp, _ = fit_shift(tr, so_tr, truth[[idx[s] for s in so_tr]])
        r_raw = aggregate(te, so_te, mode="curve")
        r_cor = aggregate(te, so_te, shift=pp)
        t = truth[[idx[s] for s in so_te]]
        print(f"     train {cam:8s} off: fit dlnb={pp[0]:+.4f} dlnc={pp[1]:+.4f} | "
              f"held-out raw {score(r_raw, t):.5f} -> {score(r_cor, t):.5f}")

    # ---- family A0: pure scale, one parameter, same two validations ----
    x0s, f0s = fit_scale(df, soils, truth)
    print(f"\nA0) pure scale  dlnb={x0s:+.4f}  (every diameter x{np.exp(x0s):.4f})")
    print(f"   in-sample {f0s:.5f}   raw {score(raw_curve, truth):.5f}")

    loo_s = []
    for s in soils:
        keep = [t for t in soils if t != s]
        xx, _ = fit_scale(df[df.soil.isin(keep)], keep, truth[[idx[t] for t in keep]])
        loo_s.append(aggregate(df[df.soil == s], [s], shift=(xx, 0.0))[0])
    print(f"   leave-one-soil-out: raw {score(np.vstack(loo_raw), truth):.5f} -> "
          f"corrected {score(np.vstack(loo_s), truth):.5f}")

    print("   leave-one-camera-out:")
    for cam in ("Edge", "Edge60fusion", "Samsung"):
        tr = df[df.camera != cam]
        te = df[df.camera == cam]
        so_tr = sorted(tr.soil.unique())
        so_te = sorted(te.soil.unique())
        xx, _ = fit_scale(tr, so_tr, truth[[idx[s] for s in so_tr]])
        r_raw = aggregate(te, so_te, mode="curve")
        r_cor = aggregate(te, so_te, shift=(xx, 0.0))
        t = truth[[idx[s] for s in so_te]]
        print(f"     held out {cam:12s}: fit dlnb={xx:+.4f} | "
              f"raw {score(r_raw, t):.5f} -> {score(r_cor, t):.5f}")

    # ---- family B: per-bin additive bias, scaled by lambda, chosen by LOOCV ----
    print("\nB) per-bin bias correction  F <- F - lam * bias,  bias = median signed error")
    for basis in ("median", "mean"):
        bias = np.median(d, axis=0) if basis == "median" else d.mean(axis=0)
        line = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            pred = np.maximum.accumulate(raw_curve - lam * bias, axis=1)
            line.append(f"lam={lam:.2f}:{score(pred, truth):.4f}")
        print(f"   in-sample ({basis}): " + "  ".join(line))

        # honest LOOCV for the whole family
        for lam in (0.0, 0.5, 1.0):
            outs = []
            for s in soils:
                keep = [t for t in soils if t != s]
                dk = raw_curve[[idx[t] for t in keep]] - truth[[idx[t] for t in keep]]
                b = np.median(dk, axis=0) if basis == "median" else dk.mean(axis=0)
                outs.append(np.maximum.accumulate(raw_curve[idx[s]] - lam * b, axis=0))
            print(f"   LOOCV  lam={lam:.2f}: {score(np.vstack(outs), truth):.4f}")


if __name__ == "__main__":
    main()
