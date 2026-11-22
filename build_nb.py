import json, os

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": s.splitlines(keepends=True)})

md("""# GRAI3 public demo -> soil PSD submission (multi-patch TTA)

**Method.** The competition host states that GRAI3 (Soranzo, *Computers and Geotechnics* 2026) is publicly
available and permitted. GRAI3 is a public Streamlit demo at
<https://huggingface.co/spaces/soranz84/grai3>. Reading its published `app.py`, the pipeline is:

1. look up the camera model + image size in `ppm.csv` to get px/mm,
2. resize by `4.6 / ppm` (LANCZOS),
3. take the **central 400 x 400** crop,
4. predict `[ln b, c]` of a Weibull grading curve

$$F(d) = 1 - \\exp\\left(-(d/b)^c\\right),$$

5. report `d10, d30, d60, Cu, Cc` and the clay/silt/sand/gravel/boulder fractions.

**Two consequences that drive this notebook.**

*Step 3 discards most of the photo.* The test photos are 4032x3024 (iPhone 14) and 5712x4284 (iPhone 16),
i.e. a ~290 mm photobox field, while a 400x400 crop at 4.6 px/mm is only an **87 x 87 mm** window. So the
demo only ever sees ~9% of each image, and one centre patch is a high-variance estimate of the bulk grading.

This notebook therefore evaluates every photo on a **3 x 3 grid of patches** (the centre plus eight
translations of +-100 mm horizontally and +-65 mm vertically). Translations are applied to the JPEG in
place with `np.roll`, keeping the pixel density and the EXIF camera tag intact, so the demo still computes
the same ppm and every patch is individually a legitimate in-distribution 87 mm view. Patch predictions are
averaged per photo, then across the photos of a sample.

*The reported diameters are enough to recover the curve exactly.* Since

$$d_p = b\\,(-\\ln(1-p))^{1/c},$$

`ln d_p` is linear in `ln(-ln(1-p))`, so a two-parameter least-squares fit through `d10, d30, d60` recovers
`b` and `c` (the three reported diameters are mutually consistent to ~0.2%). The eleven competition columns
are then `100 * F(d)` at the official sieve diameters.

Everything runs on Kaggle (CPU only, internet on). No local computation.
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

md("""## 2. Locate the competition photos and their pixel densities

`ppm.csv` values for the two test cameras are embedded so the patch translations can be converted from
millimetres to pixels; the fallback assumes the same ~290 mm photobox width the demo was calibrated on.
""")

code("""import os, glob, re, json

CANDS = ["/kaggle/input/competitions/soil-grain-size-from-photos",
         "/kaggle/input/soil-grain-size-from-photos"]
DATA = next(p for p in CANDS if os.path.isdir(p))
TEST_DIR = os.path.join(DATA, "Test_All_Photos", "Test_All_Photos")

photos = sorted(set(glob.glob(os.path.join(TEST_DIR, "*.JPG")) +
                    glob.glob(os.path.join(TEST_DIR, "*.jpg")) +
                    glob.glob(os.path.join(TEST_DIR, "*.jpeg"))))

def norm_id(s):
    s = (s.replace("\\u00dc", "Ue").replace("\\u00fc", "ue").replace("\\u00c4", "Ae")
          .replace("\\u00e4", "ae").replace("\\u00d6", "Oe").replace("\\u00f6", "oe")
          .replace("\\u00df", "ss").replace(",", "_"))
    return re.sub(r"\\s+", " ", s).strip()

def sample_id(path):
    b = os.path.basename(path)
    b = re.sub(r"\\.(jpg|jpeg)$", "", b, flags=re.I)
    b = re.sub(r"\\s*\\(\\d+\\)\\s*$", "", b)
    i = b.find("HPC")
    return norm_id(b[i:] if i >= 0 else b)

PPM = {"iPhone 14": 13.942, "iPhone 16": 19.525}

from PIL import Image
info, groups = {}, {}
for p in photos:
    im = Image.open(p)
    w, h = im.size
    cam = im.getexif().get(272)
    cam = cam.strip() if isinstance(cam, str) else None
    ppm = PPM.get(cam) or max(w, h) / 290.0
    info[os.path.basename(p)] = {"w": w, "h": h, "ppm": ppm, "cam": cam}
    groups.setdefault(sample_id(p), []).append(os.path.basename(p))

print(DATA)
print("photos:", len(photos), "samples:", len(groups))
for k, v in sorted(groups.items()):
    print(f"  {k:35s} {len(v)} photos   ppm={info[v[0]]['ppm']:.3f}   {info[v[0]]['cam']}")

json.dump({"photos": [os.path.basename(p) for p in photos], "info": info},
          open("/kaggle/working/manifest.json", "w"))
print("wrote manifest")
""")

md("""## 3. Drive the public GRAI3 demo over the patch grid

Playwright's synchronous API cannot run inside Kaggle's live asyncio loop, so the browser worker is written
to disk and executed as its own process. Streamlit's file input is hidden, so the uploader is addressed
with `state="attached"` rather than waiting for visibility. Work is ordered **offset-major**: the centre
patch for every photo is evaluated first, so an interrupted run still yields a complete centre-only
submission, and each further offset adds one patch to every photo at once.
""")

code("""WORKER = r'''
import os, re, json, time
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

SPACE    = "https://soranz84-grai3.hf.space"
MANIFEST = "/kaggle/working/manifest.json"
RAW      = "/kaggle/working/grai3_tta_raw.json"
TMP      = "/kaggle/working/patches"
os.makedirs(TMP, exist_ok=True)

man   = json.load(open(MANIFEST))
names = man["photos"]
info  = man["info"]
raw   = json.load(open(RAW)) if os.path.exists(RAW) else {}

# offsets in 4.6 px/mm screen coordinates: +-100 mm horizontal, +-65 mm vertical
OFFSETS = [(0, 0),
           (-420, 0), (420, 0), (0, -270), (0, 270),
           (-420, -270), (420, -270), (-420, 270), (420, 270)]

SRC = "/kaggle/input/competitions/soil-grain-size-from-photos/Test_All_Photos/Test_All_Photos"
if not os.path.isdir(SRC):
    SRC = "/kaggle/input/soil-grain-size-from-photos/Test_All_Photos/Test_All_Photos"

FRAC_RE = re.compile(
    r"Cl:\\s*([\\d.]+)%\\s*;\\s*Si:\\s*([\\d.]+)%\\s*;\\s*Sa:\\s*([\\d.]+)%\\s*;"
    r"\\s*Gr:\\s*([\\d.]+)%\\s*;\\s*Bo:\\s*([\\d.]+)%")

def parse_result(body):
    m = FRAC_RE.search(body)
    if not m:
        return None
    fr = {"Cl": float(m.group(1)), "Si": float(m.group(2)), "Sa": float(m.group(3)),
          "Gr": float(m.group(4)), "Bo": float(m.group(5))}
    # after the fractions the app prints d10, d30, d60 in mm, then unitless Cu, Cc.
    # Anchor on "mm" so the unitless Cu/Cc values can never be mistaken for a diameter.
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

def make_patch(name, dx, dy, dst):
    # Translate the photo in place; pixel size, density and EXIF are preserved.
    im  = Image.open(os.path.join(SRC, name))
    ex  = im.info.get("exif")
    ppm = info[name]["ppm"]
    k   = ppm / 4.6                      # screen px -> sensor px
    sx, sy = int(round(dx * k)), int(round(dy * k))
    a = np.roll(np.asarray(im.convert("RGB")), shift=(sy, sx), axis=(0, 1))
    Image.fromarray(a).save(dst, "JPEG", quality=95, exif=ex or b"")

def run_patch(page, path, timeout=180):
    page.goto(SPACE, wait_until="domcontentloaded", timeout=120000)
    inp = page.locator("input[type=file]")
    inp.wait_for(state="attached", timeout=180000)
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
                return res                      # full result, diameters present
            if best is None:
                best, t_best = res, time.time()  # fractions only, keep waiting
        # once fractions are on screen give the d-values 45 s to render
        if best is not None and time.time() - t_best > 45:
            return best
        click_confirm(page)
        page.wait_for_timeout(1500)
    return best

START  = time.time()
BUDGET = 15000

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(viewport={"width": 1100, "height": 2000})
    page = ctx.new_page()
    done = False
    for oi, (dx, dy) in enumerate(OFFSETS):
        if done:
            break
        todo = [n for n in names if f"{n}|{dx}|{dy}" not in raw]
        print(f"--- offset {oi} ({dx},{dy}) : {len(todo)} patches", flush=True)
        for i, name in enumerate(todo, 1):
            if time.time() - START > BUDGET:
                print("global budget reached, stopping early", flush=True)
                done = True
                break
            dst = os.path.join(TMP, f"{i}_{dx}_{dy}.jpg")
            res = None
            for attempt in (1, 2):
                try:
                    make_patch(name, dx, dy, dst)
                    res = run_patch(page, dst)
                except Exception as e:
                    print("  error:", type(e).__name__, str(e)[:160], flush=True)
                    res = None
                if res:
                    break
                try:
                    page.wait_for_timeout(2000)
                except Exception:
                    pass
            print(f"[o{oi} {i}/{len(todo)}] {name[:44]:44s} -> "
                  f"{'d10=%s' % res.get('d10') if res and 'd10' in res else res}", flush=True)
            if res:
                raw[f"{name}|{dx}|{dy}"] = res
                json.dump(raw, open(RAW, "w"), indent=1)
    browser.close()

print("patch results:", len(raw), "of", len(names) * len(OFFSETS), flush=True)
'''

open("/kaggle/working/grai_worker.py", "w").write(WORKER)

import subprocess, sys
r = subprocess.run([sys.executable, "/kaggle/working/grai_worker.py"],
                   capture_output=True, text=True, timeout=16200)
print(r.stdout[-14000:])
print("---- stderr ----")
print(r.stderr[-1500:])
print("worker rc:", r.returncode)
""")

md("""## 4. Recover `b`, `c` and rebuild the cumulative curves

`d10, d30, d60` are fitted jointly in log space. Curves are averaged over the patches of a photo and then
over the photos of a sample. The centre-only estimate is also rebuilt, as a check against the single-patch
run and to confirm the TTA actually moves the prediction.
""")

code("""import numpy as np
import pandas as pd

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
D_FRAC = np.array([0.002, 0.063, 2.0, 63.0])

raw = json.load(open("/kaggle/working/grai3_tta_raw.json"))
print("patch results:", len(raw))

def bc_from_diameters(fr):
    pts = [(0.1, fr.get("d10")), (0.3, fr.get("d30")), (0.6, fr.get("d60"))]
    pts = [(p, v) for p, v in pts if v and v > 0]
    if len(pts) < 2:
        return None
    # d_p = b*(-ln(1-p))^(1/c)  =>  ln d_p = ln b + (1/c)*ln(-ln(1-p))
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

def bc_from_fractions(fr):
    F = np.array([fr["Cl"], fr["Cl"] + fr["Si"], fr["Cl"] + fr["Si"] + fr["Sa"],
                  fr["Cl"] + fr["Si"] + fr["Sa"] + fr["Gr"]]) / 100.0
    from scipy.optimize import least_squares
    def resid(p):
        b, c = np.exp(p[0]), np.exp(p[1])
        return (1.0 - np.exp(-(D_FRAC / b) ** c)) - F
    sol = least_squares(resid, x0=[np.log(0.5), np.log(1.5)],
                        bounds=([np.log(1e-3), np.log(0.2)], [np.log(1e3), np.log(6.0)]))
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))

def curve_of(fr):
    bc = bc_from_diameters(fr)
    if bc is None:
        bc = bc_from_fractions(fr)
    b, c = bc
    return np.clip((1.0 - np.exp(-(SUPPORT / b) ** c)) * 100.0, 0.0, 100.0), bc

# name|dx|dy -> curve
patch = {}
for key, fr in raw.items():
    name, dx, dy = key.rsplit("|", 2)
    cv, bc = curve_of(fr)
    patch.setdefault(name, {})[(int(dx), int(dy))] = (cv, bc)

# per photo: mean over its patches
photo_curve, photo_centre, offsets_used = {}, {}, {}
for name, d in patch.items():
    photo_curve[name] = np.mean(np.vstack([v[0] for v in d.values()]), axis=0)
    offsets_used[name] = len(d)
    if (0, 0) in d:
        photo_centre[name] = d[(0, 0)][0]

def sample_mean(table):
    out = {}
    for name, cv in table.items():
        out.setdefault(sample_id(os.path.join(TEST_DIR, name)), []).append(cv)
    return {k: np.mean(np.vstack(v), axis=0) for k, v in out.items()}

tta = sample_mean(photo_curve)
cen = sample_mean(photo_centre)

print("\\npatches per photo: min %d  max %d  median %.0f"
      % (min(offsets_used.values()), max(offsets_used.values()),
         float(np.median(list(offsets_used.values())))))
print("\\ncentre-only vs multi-patch TTA (max |delta| in pp, per sample):")
for s in sorted(tta):
    d = 0.0 if s not in cen else float(np.abs(tta[s] - cen[s]).max())
    print(f"  {s:35s} n_patches={len([1 for k in patch if sample_id(os.path.join(TEST_DIR,k))==s])}  maxd={d:6.2f}")

train = pd.read_csv(os.path.join(DATA, "Training_labels_updated.csv"))
fallback = np.maximum.accumulate(train[COLS].values.mean(axis=0))
fallback[-1] = 100.0

sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
rows = []
for sid in sub["sample_id"]:
    if sid in tta:
        m = np.maximum.accumulate(np.clip(tta[sid], 0, 100))
        rows.append(m)
    else:
        print("MISSING (using training mean):", sid)
        rows.append(fallback.copy())

out = pd.DataFrame(rows, columns=COLS)
out.insert(0, "sample_id", sub["sample_id"].values)
out[COLS] = np.maximum.accumulate(np.clip(out[COLS].values, 0, 100), axis=1)
out["200"] = 100.0
out.to_csv("submission.csv", index=False)

print()
print(out.to_string(index=False))
""")

md("""## 5. Sanity checks""")

code("""import pandas as pd, numpy as np
s = pd.read_csv("submission.csv")
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
assert list(s.columns) == ["sample_id"] + COLS, s.columns
assert len(s) == 10, len(s)
assert s[COLS].isna().sum().sum() == 0
assert (s[COLS].values >= 0).all() and (s[COLS].values <= 100).all()
assert (np.diff(s[COLS].values, axis=1) >= -1e-9).all()
assert np.allclose(s["200"].values, 100.0)
print("submission.csv is valid:", s.shape)
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grai3_public_demo.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)

meta = {
    "id": "prinshudewangan/grai3-public-demo-psd",
    "title": "GRAI3 public demo PSD",
    "code_file": "grai3_public_demo.ipynb",
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
json.dump(meta, open(os.path.join(os.path.dirname(out), "kernel-metadata.json"), "w"), indent=1)
print("wrote kernel-metadata.json")
