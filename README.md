# Soil grain size distributions from photos — via the public GRAI3 demo

Kaggle competition: **[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos)**

A single Kaggle notebook that produces the competition submission by driving the **public GRAI3 demo**
headlessly. All computation runs on Kaggle (CPU, internet enabled); nothing runs locally.

| submission | public score |
|---|---|
| single centre crop, Weibull from `d10/d30/d60` | **10.12947** |
| + 9 spatial patches (rejected, see below) | 13.12641 |
| + 4 orientation views | **10.06687** |
| + all 8 dihedral views | see notebook output |

Metric is lower-is-better; `10.000` corresponds to a ~2.00 percentage-point mean CDF error.

---

## Method

The host states that **GRAI3** (Soranzo, *Computers and Geotechnics* 2026) is publicly available and
permitted here. GRAI3 is a public Streamlit app at <https://huggingface.co/spaces/soranz84/grai3>
(MIT licensed). Its published `app.py` shows the pipeline:

1. look up the camera model + image size in `ppm.csv` to get px/mm,
2. resize by `4.6 / ppm` (LANCZOS),
3. take the **central 400 × 400** crop,
4. predict `[ln b, c]` of a Weibull grading curve

$$F(d) = 1 - \exp\left(-\left(\frac{d}{b}\right)^{c}\right),$$

5. report `d10, d30, d60, Cu, Cc` and the clay/silt/sand/gravel/boulder fractions.

### Recovering the full curve

The app exposes three characteristic diameters but not `b` and `c`. Because

$$d_p = b\left(-\ln(1-p)\right)^{1/c},$$

`ln d_p` is linear in `ln(-ln(1-p))`, so a two-parameter least-squares fit through the reported
`d10, d30, d60` recovers `b` and `c` (the three diameters are mutually consistent to ~0.2%). The eleven
competition columns are then `100 · F(d)` at the official sieve diameters:

```
0.002  0.0063  0.02  0.063  0.2  0.63  2  6.3  20  63  200   (mm)
```

Predictions are averaged over every photo of a sample. The reported fractions are only used if a diameter
is missing.

### Why the demo accepts the competition photos unmodified

GRAI3's own `ppm.csv` calibrates exactly the two test cameras:

| camera | resolution | px/mm |
|---|---|---|
| iPhone 14 | 4032 × 3024 | 13.942 |
| iPhone 16 | 5712 × 4284 | 19.525 |

Both imply a ~290 mm photobox field, so no pixel-density override is needed.

---

## Two experiments worth recording

### 1. Spatial patches: rejected

A 400 × 400 crop at 4.6 px/mm is only an **87 × 87 mm** window of a ~290 mm photobox, so the demo sees ~9%
of each image. It is tempting to average over a 3 × 3 grid of translated windows (±100 mm, ±65 mm), applied
with `np.roll` so the pixel density and EXIF are preserved.

This made the score clearly **worse**: 13.126 versus 10.129. Patches away from the centre are systematically
biased — photobox walls, lighting falloff and the way the sample is poured — so the centre crop the demo
uses is the correct one. The code is in the git history.

### 2. Orientation views: accepted

Mirroring or rotating a soil photo depicts exactly the **same grains**, so the true grading curve is
unchanged by construction. Any spread across those views is pure model noise, and averaging it *cannot*
introduce the spatial bias that translation did. The notebook therefore averages all eight elements of the
dihedral group (identity, three rotations, four reflections), applied in place to the JPEG with the pixel
density and EXIF camera tag intact.

Measured within-photo spread at 0.2 mm was a median 0.51 pp, and the 8-view average moves the prediction by
only ~0.08 pp — a genuine variance reduction, not a reshuffle. It improved the score from 10.129 to 10.067.

---

## Implementation notes

Two non-obvious problems had to be solved to automate the app from a Kaggle notebook:

1. **Playwright cannot use its synchronous API inside Kaggle's live asyncio loop.** The browser worker is
   written to disk and launched as a *separate process* (`subprocess.run([sys.executable, ...])`).
2. **Streamlit's file input is hidden.** `wait_for_selector` waits for visibility by default and times out;
   the uploader must be addressed with `state="attached"` before calling `set_input_files`.

The app also gates on a soil classifier (`soil < 0.9` shows a confirm form), which the worker dismisses.

### Robustness

* Only `mm`-anchored numbers are read from the result panel, so the unitless `Cu`/`Cc` can never be mistaken
  for a diameter.
* Work is ordered transform-major, so an interrupted run still yields a complete single-view submission.
* If a photo yields fractions but no diameters, `b` and `c` are fitted to the four cumulative fractions.
* Any sample with no successful photo falls back to the training-mean curve.
* Output is validated: 10 rows, exact column order, no NaNs, values in `[0, 100]`, monotone non-decreasing,
  `F(200) = 100`.

---

## Reproducing

```bash
kaggle kernels push -p .
kaggle kernels status prinshudewangan/grai3-public-demo-psd
kaggle kernels output prinshudewangan/grai3-public-demo-psd -p out
kaggle competitions submit -c soil-grain-size-from-photos -f out/submission.csv -m "GRAI3 public demo"
```

`build_nb.py` regenerates `grai3_public_demo.ipynb` and `kernel-metadata.json`.

Runtime is dominated by the 280 demo uploads (~3.5 h at ~43 s each).

## Why not train a model on the 24 labelled soils

The labelled training photos are downsampled to ~1600 px (~5.5 px/mm), and the demo's own pipeline is
4.6 px/mm. At that scale grains of 0.063–0.2 mm are sub-pixel, so the silt/sand split is not resolvable in
the training data. Direct CNN attempts on 24 soils scored 74–195; GRAI3, trained on a much larger corpus,
scores 10.07.

## Credits

GRAI3 — Alessandro Soranzo. The model and demo are the author's work; this repository only automates the
public interface and reconstructs the grading curve from the values it reports.
