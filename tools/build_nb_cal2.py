import json, os

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": s.splitlines(keepends=True)})

md("""# GRAI3 on the labelled soils: pinning down the pixel density first

The competition ships 127 training photos whose filenames carry the soil id
(`Motorola_Edge_60_fusion_H374_01.jpg` -> `H374`), and `Training_labels_updated.csv` holds the true
cumulative passing curve of each of those 24 soils at the eleven competition sieve sizes. So the public
GRAI3 demo can be scored against ground truth on the exact competition metric, and any systematic bias in
its curves can be measured rather than guessed.

**One thing has to be settled before that measurement means anything: the physical size of the crop.**

The demo resizes by `4.6 / ppm` and then takes a 400 x 400 centre crop, i.e. a window of `400 / 4.6 =
86.96 mm` of the photo box. It reads `ppm` from its own `ppm.csv` using the camera model and the exact
image dimensions; if either is missing it falls back to `max(width, height) / 294.087` and asks the user to
confirm. The test photos are recognised (iPhone 14: 13.942, iPhone 16: 19.525). The training photos are a
different story: the three Edge 60 fusion frames are full size and keep their EXIF model, but the 69
Motorola Edge frames and the 55 Samsung A52 frames have been downscaled and stripped of EXIF, so the demo
would fall back to a much coarser density than the one the sensor actually had:

| photos | stored size | demo fallback ppm | ppm implied by the original sensor | ratio | crop becomes |
|---|---|---|---|---|---|
| Motorola Edge 60 fusion | 4096 x 2304 | 13.93 | 12.465 | 1.117 | 97.2 mm |
| Motorola Edge | 1600 x 720 / 720 x 1600 | 5.441 | 11.492 x 1600/4000 = 4.597 | 1.184 | 102.9 mm |
| Samsung A52 | 1599 x 1200 | 5.437 | 26.330 x 1599/9248 = 4.553 | 1.194 | 103.8 mm |

A 12-19 % larger window makes every grain 12-19 % smaller in the 400 x 400 input, so the demo would
report a systematically finer soil, and that error would be baked into any correction fitted on these
photos and then applied to the iPhone test photos, which are processed at their true density. It has to be
removed.

**Fix.** Before uploading, each training photo is rescaled by `294.087 * ppm_true / long_side`, i.e. by the
constant factor 0.895 (Edge 60 fusion), 0.845 (Motorola Edge) or 0.837 (Samsung A52). The demo then
guesses exactly `ppm_true` from the new dimensions and its 400 px crop covers 86.96 mm, the same physical
window it uses for the test photos.

**Check.** The three Edge 60 fusion photos (soil `H374`) are full size and still carry EXIF, so they match
`ppm.csv` exactly and the demo uses the true 12.465 ppm with no fallback. Uploading `H374_01` four ways
tests the reasoning directly:

* `A` as shipped, EXIF intact -> CSV lookup, 12.465 ppm (reference),
* `B` rescaled 0.895, EXIF intact -> fallback guess 12.466 ppm (should match `A`),
* `C` as shipped, EXIF stripped -> fallback guess 13.93 ppm (12 % too coarse a window),
* `D` rescaled 0.895, EXIF stripped -> fallback guess 12.466 ppm (should match `A` and `B`),

and the same pair on a Samsung photo (`E` as shipped -> 5.44 ppm, `F` rescaled 0.837 -> 4.55 ppm) and on
a Motorola Edge photo, which is the largest group (`G` as shipped -> 5.44 ppm, `H` rescaled 0.845 -> 4.60
ppm). The worker records the pixel density the page prints for every upload, so the values are evidence,
not assumptions. If `B` and `D` agree with `A` and `C` does not, the correction is justified.

After the experiment the notebook collects one identity view of every training photo at the corrected
scale and scores the demo against the 24 known curves. Everything runs on Kaggle (CPU only, internet on);
the model itself runs on Hugging Face's servers, so there is no local or accelerator work to do here.
""")

md("""## 1. Install a headless browser (Chromium)""")

code("""import subprocess, sys

def sh(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(((r.stdout or "") + (r.stderr or ""))[-1200:], flush=True)
    return r.returncode

sh([sys.executable, "-m", "pip", "install", "-q", "playwright"])
rc1 = sh([sys.executable, "-m", "playwright", "install", "chromium"])
rc2 = sh([sys.executable, "-m", "playwright", "install-deps", "chromium"])
print("install rc:", rc1, rc2)
""")

md("""## 2. Pair the training photos with the labels and work out each camera's scale factor""")

code("""import os, glob, re, json
from PIL import Image

CANDS = ["/kaggle/input/competitions/soil-grain-size-from-photos",
         "/kaggle/input/soil-grain-size-from-photos"]
DATA = next(p for p in CANDS if os.path.isdir(p))
TRAIN_DIR = os.path.join(DATA, "Training-All_Photos_updated", "Training-All_Photos_updated")
if not os.path.isdir(TRAIN_DIR):
    TRAIN_DIR = os.path.join(DATA, "Training-All_Photos_updated")

photos = sorted(set(glob.glob(os.path.join(TRAIN_DIR, "*.JPG")) +
                    glob.glob(os.path.join(TRAIN_DIR, "*.jpg")) +
                    glob.glob(os.path.join(TRAIN_DIR, "*.jpeg"))))

def soil_of(path):
    b = os.path.basename(path)
    b = re.sub(r"\\.(jpg|jpeg)$", "", b, flags=re.I)
    b = re.sub(r"\\s*\\(\\d+\\)\\s*$", "", b)
    b = re.sub(r"_\\d+$", "", b)
    return b.split("_")[-1]

# long side of the original sensor frame and its calibrated pixel density (the demo's own ppm.csv)
# order matters: the Edge 60 fusion key must be tested before the generic Motorola Edge key
SENSOR = {"Motorola_Edge_60_fusion": (4096, 12.465),
          "Motorola_Edge": (4000, 11.492),
          "Samsung_A52": (9248, 26.330)}

def camera_of(name):
    for k in SENSOR:
        if name.startswith(k):
            return k
    return None

def scale_for(name, long_side):
    cam = camera_of(name)
    if cam is None:
        return 1.0, None
    long_orig, ppm = SENSOR[cam]
    ppm_true = ppm * long_side / long_orig
    return 294.087 * ppm_true / long_side, ppm_true      # == 294.087 * ppm / long_orig

import pandas as pd
lab = pd.read_csv(os.path.join(DATA, "Training_labels_updated.csv"))
known = set(lab["sample_id"])

info, groups = {}, {}
for p in photos:
    n = os.path.basename(p)
    im = Image.open(p)
    long_side = max(im.size)
    s, ppm_true = scale_for(n, long_side)
    info[n] = {"size": list(im.size), "soil": soil_of(p), "scale": round(s, 5),
               "ppm_true": None if ppm_true is None else round(ppm_true, 4),
               "exif": im.getexif().get(272)}
    groups.setdefault(soil_of(p), []).append(n)

print(DATA)
print("training photos:", len(photos), "soils:", len(groups))
bad = sorted(k for k in groups if k not in known)
assert not bad, bad
print("soils with no photo:", sorted(known - set(groups)))
nocam = sorted(k for k in info if camera_of(k) is None)
assert not nocam, nocam

from collections import Counter
print("\\nper camera:")
for cam in SENSOR:
    rows = [v for k, v in info.items() if camera_of(k) == cam]
    print(f"  {cam:26s} {len(rows):3d} photos  scale={rows[0]['scale']:.4f}  "
          f"ppm_true~{rows[0]['ppm_true']:.3f}  (fallback would be "
          f"{max(rows[0]['size'])/294.087:.3f})")

exif_photos = [k for k, v in info.items() if v["exif"]]
assert len(exif_photos) == 3, exif_photos
print("\\nphotos that still carry an EXIF camera model:", exif_photos)

# the four ways of presenting H374_01, and the Samsung pair
h374 = next(k for k in exif_photos)
sam = next(k for k in info if camera_of(k) == "Samsung_A52")
mot = next(k for k in info if camera_of(k) == "Motorola_Edge")
cases = [
    {"tag": "A_as_shipped_exif",   "photo": h374, "scale": 1.0, "keep_exif": True},
    {"tag": "B_rescaled_exif",     "photo": h374, "scale": info[h374]["scale"], "keep_exif": True},
    {"tag": "C_as_shipped_noexif", "photo": h374, "scale": 1.0, "keep_exif": False},
    {"tag": "D_rescaled_noexif",   "photo": h374, "scale": info[h374]["scale"], "keep_exif": False},
    {"tag": "E_samsung_as_shipped","photo": sam,  "scale": 1.0, "keep_exif": False},
    {"tag": "F_samsung_rescaled",  "photo": sam,  "scale": info[sam]["scale"], "keep_exif": False},
    {"tag": "G_edge_as_shipped",   "photo": mot,  "scale": 1.0, "keep_exif": False},
    {"tag": "H_edge_rescaled",     "photo": mot,  "scale": info[mot]["scale"], "keep_exif": False},
]
print("\\nexperiment cases:")
for c in cases:
    print("  ", c)

json.dump({"src": TRAIN_DIR,
           "photos": [os.path.basename(p) for p in photos],
           "info": info,
           "soil": {os.path.basename(p): soil_of(p) for p in photos},
           "cases": cases},
          open("/kaggle/working/exp_manifest.json", "w"))
print("\\nwrote exp_manifest.json")
""")

md("""## 3. Drive the demo: the scale experiment, then one view of every training photo

The browser worker runs as its own process (Playwright's sync API cannot live inside Kaggle's asyncio
loop). The uploader is hidden, so it is addressed with `state="attached"`; two `Confirm` buttons exist (the
pixel-density form and the soil-classifier gate) and the worker clicks whichever appears. Results are
written after every single upload, so a run that is cut short still leaves usable data.
""")

code("""WORKER = r'''
import os, re, json, time
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

SPACE    = "https://soranz84-grai3.hf.space"
MANIFEST = "/kaggle/working/exp_manifest.json"
RAW      = "/kaggle/working/train_raw.json"
EXP      = "/kaggle/working/scale_experiment.json"
TMP      = "/kaggle/working/exp_patches"
os.makedirs(TMP, exist_ok=True)

man    = json.load(open(MANIFEST))
info   = man["info"]
names  = man["photos"]
SRC    = man["src"]
cases  = man["cases"]
raw    = json.load(open(RAW)) if os.path.exists(RAW) else {}
exp    = json.load(open(EXP)) if os.path.exists(EXP) else {}

FRAC_RE = re.compile(
    r"Cl:\\s*([\\d.]+)%\\s*;\\s*Si:\\s*([\\d.]+)%\\s*;\\s*Sa:\\s*([\\d.]+)%\\s*;"
    r"\\s*Gr:\\s*([\\d.]+)%\\s*;\\s*Bo:\\s*([\\d.]+)%")
PPM_RE = re.compile(r"pixel density\\s*([0-9.]+)\\s*PPM", re.I)

def parse_result(body):
    m = FRAC_RE.search(body)
    if not m:
        return None
    fr = {"Cl": float(m.group(1)), "Si": float(m.group(2)), "Sa": float(m.group(3)),
          "Gr": float(m.group(4)), "Bo": float(m.group(5))}
    pm = PPM_RE.search(body)
    fr["ppm"] = float(pm.group(1)) if pm else None
    tail = body[m.end():]
    nums = [float(x) for x in re.findall(r":\\s*([0-9]+(?:\\.[0-9]+)?)\\s*mm", tail)]
    if len(nums) < 3:
        nums = [float(x) for x in re.findall(r":\\s*([0-9]+\\.[0-9]+)", tail)][:3]
    if len(nums) >= 3:
        d10, d30, d60 = nums[0], nums[1], nums[2]
        if 0 < d10 <= d30 <= d60:
            fr["d10"], fr["d30"], fr["d60"] = d10, d30, d60
    return fr

def click_confirm(page):
    for b in page.query_selector_all("button"):
        try:
            if b.inner_text().strip().lower() == "confirm":
                b.click(); return True
        except Exception:
            pass
    return False

def make_image(name, scale, keep_exif, dst):
    im = Image.open(os.path.join(SRC, name))
    ex = None
    if keep_exif:
        ex = im.info.get("exif") or im.getexif().tobytes() or None
    if abs(scale - 1.0) > 0.002:
        w, h = im.size
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                       Image.LANCZOS)
    im.convert("RGB").save(dst, "JPEG", quality=95, exif=ex or b"")

def space_ready(page, timeout=60):
    try:
        page.goto(SPACE, wait_until="domcontentloaded", timeout=timeout * 1000)
        page.locator("input[type=file]").wait_for(state="attached", timeout=timeout * 1000)
        return True
    except Exception as e:
        print("  demo not answering:", type(e).__name__, str(e)[:110], flush=True)
        return False

def wait_for_space(page, max_wait=1800):
    t0, delay, n = time.time(), 15, 0
    while time.time() - t0 < max_wait:
        n += 1
        if space_ready(page, timeout=90):
            print(f"  demo ready after {int(time.time() - t0)} s / {n} tries", flush=True)
            return True
        try:
            page.wait_for_timeout(delay * 1000)
        except Exception:
            pass
        delay = min(delay * 2, 120)
    return False

def run_patch(page, path, timeout=180):
    page.goto(SPACE, wait_until="domcontentloaded", timeout=90000)
    inp = page.locator("input[type=file]")
    inp.wait_for(state="attached", timeout=90000)
    inp.set_input_files(path)
    t0 = time.time()
    best, t_best = None, None
    while time.time() - t0 < timeout:
        try:
            body = page.inner_text("body")
        except Exception:
            body = ""
        res = parse_result(body)
        if res is not None:
            if "d10" in res:
                return res
            if best is None:
                best, t_best = res, time.time()
        if best is not None and time.time() - t_best > 45:
            return best
        click_confirm(page)
        page.wait_for_timeout(1500)
    return best

def attempt(page, name, scale, keep_exif, dst, tries=3):
    for k in range(1, tries + 1):
        try:
            make_image(name, scale, keep_exif, dst)
            res = run_patch(page, dst)
            if res:
                return res
        except Exception as e:
            print(f"  attempt {k} failed: {type(e).__name__} {str(e)[:140]}", flush=True)
        if k < tries:
            wait_for_space(page, max_wait=1200)
    return None

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(viewport={"width": 1100, "height": 2000})
    page = ctx.new_page()

    print("=== pre-flight: waiting for the demo to answer ===", flush=True)
    if not wait_for_space(page, max_wait=1800):
        raise RuntimeError("the demo never became ready")

    print("=== scale experiment ===", flush=True)
    for c in cases:
        if c["tag"] in exp:
            print("  already done:", c["tag"], exp[c["tag"]], flush=True)
            continue
        dst = os.path.join(TMP, "exp_%s.jpg" % c["tag"])
        r = attempt(page, c["photo"], c["scale"], c["keep_exif"], dst)
        exp[c["tag"]] = r
        json.dump(exp, open(EXP, "w"), indent=1)
        print(f"  {c['tag']:24s} scale={c['scale']:.4f} -> {r}", flush=True)

    print("=== collection ===", flush=True)
    START  = time.time()
    BUDGET = 10800
    todo = [n for n in names if f"{n}|0|0" not in raw]
    print("photos to collect:", len(todo), flush=True)
    fails = 0
    for i, name in enumerate(todo, 1):
        if time.time() - START > BUDGET:
            print("budget reached", flush=True)
            break
        dst = os.path.join(TMP, f"{i}_0.jpg")
        r = attempt(page, name, info[name]["scale"], True, dst)
        print(f"[{i}/{len(todo)}] {name[:48]:48s} scale={info[name]['scale']:.3f} "
              f"ppm={r.get('ppm') if r else None} -> "
              f"{'d10=%s' % r.get('d10') if r and 'd10' in r else r}", flush=True)
        if r:
            raw[f"{name}|0|0"] = r
            json.dump(raw, open(RAW, "w"), indent=1)
            fails = 0
        else:
            fails += 1
            print(f"  {fails} consecutive failures -> waiting for the demo", flush=True)
            wait_for_space(page, max_wait=2400)
    browser.close()

print("collected:", len(raw), "of", len(names), flush=True)
'''

open("/kaggle/working/grai_worker.py", "w").write(WORKER)

import subprocess, sys, shutil
try:
    r = subprocess.run([sys.executable, "/kaggle/working/grai_worker.py"],
                       capture_output=True, text=True, timeout=14400)
    print(r.stdout[-16000:])
    print("---- stderr ----")
    print(r.stderr[-1200:])
    print("worker rc:", r.returncode)
except subprocess.TimeoutExpired as e:
    print("worker hit the notebook timeout; partial results are kept")
    out = e.stdout or ""
    print((out.decode("utf-8", "replace") if isinstance(out, bytes) else out)[-4000:])
finally:
    shutil.rmtree("/kaggle/working/exp_patches", ignore_errors=True)
    print("removed patch images")
""")

md("""## 4. Score the demo against the known curves

The metric is the sum of `|dCDF| * dlog10(d)` over the ten sieve intervals; with these eleven sieves every
interval is ~0.5 wide, so the score is ~5x the mean absolute error in percentage points of cumulative
passing. A score of 10.03 means an average miss of about 2 percentage points.
""")

code("""import numpy as np, pandas as pd, json, os

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
W = np.diff(np.log10(SUPPORT))

man = json.load(open("/kaggle/working/exp_manifest.json"))
exp = json.load(open("/kaggle/working/scale_experiment.json"))
raw = json.load(open("/kaggle/working/train_raw.json"))
print("scale experiment:")
for k, v in exp.items():
    if v:
        print(f"  {k:24s} ppm={v.get('ppm')}  d10={v.get('d10')}  d30={v.get('d30')}  d60={v.get('d60')}")
    else:
        print(f"  {k:24s} FAILED")

print("\\nviews collected:", len(raw), "of", len(man["photos"]))

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

rows = []
for key, fr in raw.items():
    name = key.rsplit("|", 2)[0]
    bc = bc_from_diameters(fr)
    if bc is None:
        print("could not fit:", key, fr)
        continue
    cam = ("Edge60fusion" if name.startswith("Motorola_Edge_60_fusion")
           else "Edge" if name.startswith("Motorola_Edge") else "Samsung")
    rows.append({"photo": name, "soil": man["soil"][name], "camera": cam,
                 "ppm_used": fr.get("ppm"),
                 "b": bc[0], "c": bc[1],
                 "Cl": fr["Cl"], "Si": fr["Si"], "Sa": fr["Sa"], "Gr": fr["Gr"], "Bo": fr["Bo"]})

df = pd.DataFrame(rows)
if len(df):
    df.to_csv("/kaggle/working/train_views.csv", index=False)
    print("per-view rows:", len(df), "photos:", df.photo.nunique(), "soils:", df.soil.nunique())

    lab = pd.read_csv(os.path.join(DATA, "Training_labels_updated.csv")).set_index("sample_id")
    soils = sorted(df.soil.unique())
    truth = np.maximum.accumulate(lab.loc[soils, COLS].values.astype(float), axis=1)

    def curve(b, c):
        return np.clip((1.0 - np.exp(-((SUPPORT / b) ** c))) * 100.0, 0.0, 100.0)

    pred = np.vstack([curve(b, c) for b, c in zip(df.b, df.c)])
    agg = np.maximum.accumulate(np.vstack([pred[(df.soil == s).values].mean(axis=0) for s in soils]), axis=1)
    d = np.abs(agg - truth)
    print(f"\\nmean |dCDF|: {d.mean():.3f} pp")
    print(f"competition metric on the {len(soils)} labelled soils: {float((d[:, 1:] * W).sum() / len(soils)):.5f}")
    print("\\nper soil:")
    for i, s in enumerate(soils):
        print(f"  {s}  n={int((df.soil == s).sum())}  score={float((d[i, 1:] * W).sum()):6.2f}  "
              f"max={d[i].max():6.2f}")
    print("\\nper bin (pred - true):")
    for j, s in enumerate(COLS):
        print(f"  {s:>7s}  signed {(agg - truth)[:, j].mean():+7.2f}   abs {d[:, j].mean():6.2f}")
    np.save("/kaggle/working/train_agg.npy", agg)
    np.save("/kaggle/working/train_truth.npy", truth)
    print("\\nwrote train_views.csv")
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "grai3_calib2.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)

meta = {
    "id": "prinshudewangan/grai3-scale-and-calibration",
    "title": "GRAI3 scale and calibration",
    "code_file": "grai3_calib2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": ["soil-grain-size-from-photos"],
    "model_sources": [],
}
json.dump(meta, open(os.path.join(here, "kernel-metadata-calib2.json"), "w"), indent=1)
print("wrote kernel-metadata-calib2.json")
