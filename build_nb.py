import json, os

cells = []

def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": s.splitlines(keepends=True)})

md("""# GRAI3 public demo -> soil PSD submission (orientation TTA)

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
i.e. a ~290 mm photobox field, while a 400x400 crop at 4.6 px/mm is only an **87 x 87 mm** window, so the
demo only ever sees ~9% of each image.

**A failed experiment, reported honestly.** Averaging that centre window with eight translated windows
(+-100 mm, +-65 mm) was tried first. It made the public score clearly *worse* (13.126 vs 10.129): patches
away from the centre are systematically biased, presumably by the photobox walls, the lighting falloff and
the way the sample is poured, so the centre crop the demo uses is the correct one. That version is kept in
the git history, not used here.

**What this notebook does instead.** It averages over *orientation* transforms of the very same pixels: all
sixteen elements of the dihedral group (rotations by multiples of 45 degrees, and their reflections). A
mirror or rotation of a soil photo depicts exactly the same grains, so the true grading curve is unchanged
by construction: any spread across these views is pure model noise, and averaging it cannot introduce the
spatial bias that translation did. Transforms are applied to the JPEG in place, keeping the pixel density
and the EXIF camera tag intact (the demo's ppm lookup accepts the image either way round), so every view is
a legitimate in-distribution 87 mm crop.

The eight axis-aligned views (identity, 90/180/270 degree rotations and their mirrors) are exact pixel
permutations; the eight 45-degree views resample the full-resolution JPEG bicubically before the demo's
LANCZOS resize, which is a far smaller kernel than the 0.24-0.33x downsample that follows. The central
87 x 87 mm window lies well inside the rotated frame, so no black fill ever reaches the crop.

The eight axis-aligned views are read from the previous run's published JSON
(`data/grai3_d8_views.json` in the repository), so the notebook only drives the demo for the eight new
views. If that file cannot be fetched, the notebook falls back to collecting all sixteen itself.

*Why 16 and not 8.* Splitting the eight axis-aligned views into two halves of four showed that the halves
still disagree by 0.02-0.16 pp on the cumulative curve (SE of the 8-view mean: 0.031 pp). That residual is
pure view noise, and it is what the last 0.03 of public score is made of, so the view set is doubled here.

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

# The eight axis-aligned views were collected and published by the previous run.
# Seeding them here means this run only has to collect the eight 45-degree views;
# if the URL is unreachable the notebook falls back to collecting all sixteen.
SEED_URL = ("https://raw.githubusercontent.com/prinshu756/grai3-public-demo/"
            "main/data/grai3_d8_views.json")
try:
    import urllib.request
    with urllib.request.urlopen(SEED_URL, timeout=90) as f:
        seed = json.load(f)
    json.dump(seed, open("/kaggle/working/grai3_tta_raw.json", "w"))
    print("seeded", len(seed), "precomputed views from", SEED_URL)
except Exception as e:
    print("seed unavailable, collecting all 16 views:", type(e).__name__, str(e)[:120])
""")

md("""## 3. Drive the public GRAI3 demo over the orientation views

Playwright's synchronous API cannot run inside Kaggle's live asyncio loop, so the browser worker is written
to disk and executed as its own process. Streamlit's file input is hidden, so the uploader is addressed
with `state="attached"` rather than waiting for visibility. Work is ordered **transform-major**: the
identity view of every photo is evaluated first, so an interrupted run still yields a complete
single-view submission, and each further transform adds one view to every photo at once.
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

# sixteen elements of the dihedral group of the square-sampled circle:
# 0-7  : axis-aligned views (identity, 90/180/270 rotations and four mirrors)
# 8-15 : the same eight views pre-rotated by 45 degrees
# 0-7 are exact pixel permutations, so a partial run is already the previous 8-view set.
TRANSFORMS = list(range(16))

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

def make_transform(name, t, dst):
    # Mirror/rotate in place; the pixel set is unchanged, EXIF is preserved.
    im = Image.open(os.path.join(SRC, name))
    ex = im.info.get("exif")
    a = np.asarray(im.convert("RGB"))
    if t == 1:
        a = a[:, ::-1]                  # horizontal flip
    elif t == 2:
        a = a[::-1, :]                  # vertical flip
    elif t == 3:
        a = a[::-1, ::-1]               # 180 degree rotation
    elif t == 4:
        a = np.rot90(a, 1)              # 90 degree rotation (dims swap)
    elif t == 5:
        a = np.rot90(a, 3)              # 270 degree rotation (dims swap)
    elif t == 6:
        a = a.transpose(1, 0, 2)        # reflection across the main diagonal
    elif t == 7:
        a = a.transpose(1, 0, 2)[::-1, ::-1]   # reflection across the anti-diagonal
    elif t >= 8:
        k, f = (t - 8) // 2, (t - 8) % 2        # 45/135/225/315 degrees, optionally mirrored
        img = Image.fromarray(np.ascontiguousarray(a))
        if f:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(45.0 + 90.0 * k, resample=Image.BICUBIC, expand=False)
        a = np.asarray(img)
    Image.fromarray(np.ascontiguousarray(a)).save(dst, "JPEG", quality=95, exif=ex or b"")

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
BUDGET = 36000

with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    ctx = browser.new_context(viewport={"width": 1100, "height": 2000})
    page = ctx.new_page()
    done = False
    for oi, t in enumerate(TRANSFORMS):
        if done:
            break
        todo = [n for n in names if f"{n}|{t}|0" not in raw]
        print(f"--- transform {oi} (t={t}) : {len(todo)} views", flush=True)
        fails = 0
        for i, name in enumerate(todo, 1):
            if time.time() - START > BUDGET:
                print("global budget reached, stopping early", flush=True)
                done = True
                break
            dst = os.path.join(TMP, f"{i}_{t}.jpg")
            res = None
            for attempt in (1, 2):
                try:
                    make_transform(name, t, dst)
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
            print(f"[t{oi} {i}/{len(todo)}] {name[:44]:44s} -> "
                  f"{'d10=%s' % res.get('d10') if res and 'd10' in res else res}", flush=True)
            if res:
                raw[f"{name}|{t}|0"] = res
                json.dump(raw, open(RAW, "w"), indent=1)
                fails = 0
            else:
                fails += 1
                if fails >= 4:
                    print("  four consecutive failures: abandoning this transform", flush=True)
                    break
    browser.close()

print("view results:", len(raw), "of", len(names) * len(TRANSFORMS), flush=True)
'''

open("/kaggle/working/grai_worker.py", "w").write(WORKER)

import subprocess, sys, shutil
r = subprocess.run([sys.executable, "/kaggle/working/grai_worker.py"],
                   capture_output=True, text=True, timeout=39000)
print(r.stdout[-14000:])
print("---- stderr ----")
print(r.stderr[-1500:])
print("worker rc:", r.returncode)
shutil.rmtree(TMP, ignore_errors=True)   # keep the kernel output small: only the JSON is needed
print("removed patch images")
""")

md("""## 4. Recover `b`, `c` and rebuild the cumulative curves

`d10, d30, d60` are fitted jointly in log space. Curves are averaged over the orientation views of a photo
and then over the photos of a sample. The identity-only estimate is also rebuilt, as a check against the
single-view run and to confirm that averaging actually moves the prediction.
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

# name|t|dy -> curve
patch = {}
for key, fr in raw.items():
    name, t, dy = key.rsplit("|", 2)
    cv, bc = curve_of(fr)
    patch.setdefault(name, {})[(int(t), int(dy))] = (cv, bc)

def photo_mean(table, tset):
    out = {}
    for name, d in table.items():
        vs = [v[0] for k, v in d.items() if k[0] in tset]
        if vs:
            out[name] = np.mean(np.vstack(vs), axis=0)
    return out

def sample_mean(table):
    out = {}
    for name, cv in table.items():
        out.setdefault(sample_id(os.path.join(TEST_DIR, name)), []).append(cv)
    return {k: np.mean(np.vstack(v), axis=0) for k, v in out.items()}

views = {}
for t in range(16):
    views[t] = sum(1 for d in patch.values() for k in d if k[0] == t)
print("views collected per transform:", views)
print("total views:", sum(views.values()), " photos:", len(patch))

tta8  = sample_mean(photo_mean(patch, set(range(8))))
tta16 = sample_mean(photo_mean(patch, set(range(16))))

print("\\n8-view vs 16-view sample curves (max |delta| in pp):")
for s in sorted(tta16):
    d = float(np.abs(tta16[s] - tta8[s]).max()) if s in tta8 else float("nan")
    print(f"  {s:35s} maxd={d:6.3f}")

train = pd.read_csv(os.path.join(DATA, "Training_labels_updated.csv"))
fallback = np.maximum.accumulate(train[COLS].values.mean(axis=0))
fallback[-1] = 100.0

sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

def frame(tta, path):
    rows = []
    for sid in sub["sample_id"]:
        if sid in tta:
            rows.append(np.maximum.accumulate(np.clip(tta[sid], 0, 100)))
        else:
            print("MISSING (using training mean):", sid)
            rows.append(fallback.copy())
    out = pd.DataFrame(rows, columns=COLS)
    out.insert(0, "sample_id", sub["sample_id"].values)
    out[COLS] = np.maximum.accumulate(np.clip(out[COLS].values, 0, 100), axis=1)
    out["200"] = 100.0
    out.to_csv(path, index=False)
    return out

out = frame(tta16, "submission.csv")
frame(tta8, "submission_8view.csv")

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
