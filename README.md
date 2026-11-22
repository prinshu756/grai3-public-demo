# Soil grain size distributions from photos — via the public GRAI3 demo

Kaggle competition: **[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos)**

This repository contains a single Kaggle notebook that produces the competition submission by driving the
**public GRAI3 demo** headlessly. All computation runs on Kaggle (CPU, internet enabled); nothing runs locally.

---

## Method

The competition asks for the cumulative grain-size distribution of 10 soil samples, each photographed
several times. The host states that **GRAI3** (Soranzo, *Computers and Geotechnics* 2026) is publicly
available and permitted for use in this competition.

GRAI3 is a public Streamlit app at <https://huggingface.co/spaces/soranz84/grai3> (MIT licensed). Internally it

1. resamples the input photo to **4.6 px/mm**,
2. takes the centre **400 × 400** crop (an ~87 mm field of view),
3. predicts the two parameters `b`, `c` of a Weibull grading curve

$$F(d) = 1 - \exp\left(-\left(\frac{d}{b}\right)^{c}\right),$$

and reports the resulting clay/silt/sand/gravel/boulder fractions plus `d10`, `d30`, `d60`, `Cu`, `Cc`.

### Recovering the full curve

The app exposes three characteristic diameters but not `b` and `c` directly. Because

$$d_p = b\left(-\ln(1-p)\right)^{1/c},$$

taking logs gives a straight line in `(ln(-ln(1-p)), ln d_p)`. A two-parameter least-squares fit through the
reported `d10`, `d30`, `d60` recovers `b` and `c` exactly (the reported values are mutually consistent to
better than 0.2%). The eleven competition columns are then `100 · F(d)` at the official sieve diameters

```
0.002  0.0063  0.02  0.063  0.2  0.63  2  6.3  20  63  200   (mm)
```

Predictions are averaged over every photo of a sample.

### Why the demo accepts the competition photos unmodified

GRAI3's own `ppm.csv` calibrates exactly the two cameras used for the test set:

| camera | resolution | px/mm |
|---|---|---|
| iPhone 14 | 4032 × 3024 | 13.942 |
| iPhone 16 | 5712 × 4284 | 19.525 |

so the pixel-density override the demo offers is not needed — the app resamples to 4.6 px/mm by itself.

---

## Implementation notes

Two non-obvious problems had to be solved to automate the app from a Kaggle notebook:

1. **Playwright cannot use its synchronous API inside Kaggle's live asyncio loop.** The browser worker is
   therefore written to disk and launched as a *separate process* (`subprocess.run([sys.executable, ...])`).
2. **Streamlit's file input is hidden.** `wait_for_selector` waits for visibility by default and times out;
   the uploader must be addressed with `state="attached"` before calling `set_input_files`.

The notebook installs Chromium, uploads all 35 test photos one at a time (~55 s each), caches every result
to JSON, then reconstructs the curves and validates the submission.

### Robustness

* Only `mm`-anchored numbers are read from the result panel, so the unitless `Cu`/`Cc` values can never be
  mistaken for a diameter.
* If a photo yields fractions but no diameters, `b` and `c` are fitted to the four cumulative fractions
  (`Cl`, `Cl+Si`, `Cl+Si+Sa`, `Cl+Si+Sa+Gr`) instead.
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

---

## Result

Public leaderboard score: **TBD** (metric is lower-is-better; `10.000` corresponds to roughly a 2 percentage
point mean CDF error).

## Credits

GRAI3 — Alessandro Soranzo. The model and demo are the author's work; this repository only automates the
public interface and reconstructs the grading curve from the values it reports.
