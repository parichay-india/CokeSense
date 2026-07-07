# CokeSense

Calibrated coke-quality early warning for the blast-furnace route: a hybrid
soft sensor for **CRI** (coke reactivity index) and **CSR** (coke strength
after reaction) with a **rolling conformal** uncertainty band, evaluated on a
decade of daily records (1,251 rows, Dec 2012 - Mar 2023) from an operating
integrated steel plant and served from a 37 KB model in about a millisecond.

Live dashboard: https://cokequalitybsl.streamlit.app/

> **Double-blind notice.** The accompanying paper is under anonymous review
> at ICDCN 2027. Keep this repository **private** until the camera-ready
> version is due; making it public earlier links the anonymized submission to
> its authors.

## What is in here

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard implementing the paper's serving loop (hybrid ridge + rolling conformal band) |
| `models/` | Trained artifacts: `hybrid_ridge_CRI.joblib`, `hybrid_ridge_CSR.joblib`, `meta.json` (quantiles, residual buffer, defaults, ranges) |
| `notebooks/CokeSense_Replication.ipynb` | One notebook that reproduces every table, figure, and audited number of the paper, and regenerates `models/` |
| `data/Coke_Oven_Data_Set.xlsx` | The de-identified daily plant workbook (nine sheets; the `Coke` sheet drives everything) |
| `requirements.txt` | Pinned versions; the joblib artifacts load only under scikit-learn 1.8.0 |

## Quickstart

Run the dashboard locally:

    pip install -r requirements.txt
    streamlit run app.py

Reproduce the paper (Colab or local Jupyter): open
`notebooks/CokeSense_Replication.ipynb` and run all cells. With
`FULL_RUN = True` (default) the run takes about 35 minutes on a free Colab
CPU and ends with a numeric audit against the manuscript; figures land in
`figs/` with the exact filenames the LaTeX source expects.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (private is fine; Streamlit Cloud can read
   private repositories you authorize).
2. On https://share.streamlit.io create or edit the app, point it at this
   repository, branch `main`, main file `app.py`.
3. For the existing `cokequalitybsl` app, replacing the repository contents
   through the GitHub web interface (Add file -> Upload files, overwrite) and
   rebooting the app from the Streamlit Cloud console is sufficient; the
   pinned `requirements.txt` triggers a clean dependency rebuild.

## Method in one paragraph

Point estimates come from a ridge pipeline over eleven coke-side features
(eight raw measurements plus three engineered terms) joined to the last
confirmed laboratory value and its seven-day mean; the interval is a split-
conformal band whose half-width is recomputed daily from the most recent 100
absolute residuals with confirmed labels. Replayed chronologically over
2019-2023 this configuration reached MAE 0.466 (CRI) and 0.627 (CSR), within
11% and 6% of the replicate-noise floor of the laboratory test itself, with
empirical coverage of 90.4% and 92.4% at the nominal 90% level. Details,
ablations, and the sixteen-model benchmark are in the paper and the notebook.

## Retraining cadence

Rerun the notebook's deployment cell (Section 11) whenever the operating
regime or the blend strategy changes, or on a fixed quarterly schedule;
commit the refreshed `models/` directory and reboot the app.

## Citation

The paper is under review; a citation block and DOI will replace this line
upon acceptance.
