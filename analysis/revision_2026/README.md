# Revision analyses — CURRENT-BIOLOGY-D-25-00716 (2026)

Scripts, figures, and derived data produced while answering the reviewers of
*"Collective flow of circadian clock information in honeybee colonies"*. Two
things came out of this work and are reflected across the repo:

1. **A day/night analysis** of the age-vs-distance distribution (Reviewer 2,
   Minor 2) → **Supplementary Note 5** + `speed_transfer_paper/daynight_age_distance.pdf`.
2. **A correction to the 2016 (colony A) nest-entrance coordinate** used to
   compute distance-to-entrance → fixes **Supplementary Note 2** and **Fig S1e**.

---

## The 2016 entrance-coordinate correction (important)

`path_settings.set_parameters(2016, …)` originally used `exit_pos = (0, 250)`.
The 2016 recording **stitches two cameras per side** into one data-level frame
(four cameras for the comb); `(0, 250)` pointed at the wrong/empty comb corner,
which **inverted colony A's distance-to-entrance relationships** and produced the
"colony inversion / different hive geometry" wording in an earlier draft of
Note 2.

The true 2016 entrance is the **bottom-right comb corner, `(346, 270)`**, verified
three independent ways on the actual data (`entrance_diag.py`, run against the
`filter_overlap` positions in `data/dist_exit_df_side0_2016_CORRECTED.csv`):

| metric vs. distance | at `(0,250)` | at `(346,270)` | model expects |
|---|---|---|---|
| phase   | r = −0.20 ✗ | r = +0.34 ✓ | phase rises with distance |
| age     | r = +0.42 ✗ | r = −0.67 ✓ | older near the entrance |
| R²      | r = +0.56 ✗ | r = −0.79 ✓ | stronger rhythm near entrance |

`(0,250)` has the wrong sign on all three; `(346,270)` is correct. Corrected,
colony A matches colony B. **2019's `(5, 264)` was already correct.**

Two secondary findings, both handled in the manuscript:
- **Note 2 was WLS, the rest of the paper is OLS.** The `correlation_of_*` scripts
  used an inverse-variance × inverse-sampling weighting that inflates R² and can
  flip signs. Note 2 is now reported in plain **OLS** (see `note2_ols_stats.py`).
- The 2016 stitch has a **thin ~1-unit density-overlap stripe at x≈180** where the
  two cameras meet (`full_seam.py`) — benign: positions are coherent (phase tracks
  distance cleanly), only counts are locally doubled.

### Changes applied elsewhere in the repo
- `path_settings.py` — 2016 `exit_pos → (346, 270)` (commented).
- `analysis/aggregate_cosinor_estimates_per_hive_position.py` — vectorized the
  distance computation (identical result) and `reset_index()` before saving so
  `x_pos_start_focal` / `y_pos_start_focal` are kept (needed for Fig S1e and the
  entrance diagnostics).
- `speed_transfer_paper/main_revised.tex` — Note 2 → OLS with corrected numbers,
  "inversion" sentence removed; Fig S1e regenerated (`figure_S1e_corrected.pdf`);
  Supplementary Note 5 (day/night) added, colony-B-only.

---

## Scripts

Run the **derived-data** scripts from this directory — they read the committed
`data/` and (for the plots) write into `speed_transfer_paper/`. They need only
`pandas`/`numpy`/`matplotlib`, except `note2_ols_stats.py` which needs
`statsmodels` (use the horus venv; see below).

| script | what it does | inputs |
|---|---|---|
| `note2_ols_stats.py --year {2016,2019}` | corrected **OLS** Note 2 regressions (dist∼age, dist∼phase, dist∼R², phase∼age) | `data/dist_exit_df_side0_*.csv` |
| `entrance_diag.py` | count-weighted correlation grid-search that **locates the entrance** and compares `(0,250)` vs `(346,270)` | `data/dist_exit_df_side0_2016_CORRECTED.csv` |
| `plot_s1e.py` | regenerates **Fig S1e** (colony A) from the corrected `dist_exit_df` | `data/…2016_CORRECTED.csv` |
| `plot_daynight_colonyB.py` | regenerates the **SI Note 5** figure | `data/daynight_2019.csv` |

The **raw-data** scripts below stream the ~18–41 GB interaction CSVs
(`interactions_side0_{year}.csv`) and must run where those live (trove / the
horus setup). Point them at the data with `SPEEDTRANSFER_INTERACTIONS=…` and
choose an output dir with `REVISION_OUT=…`.

| script | what it does |
|---|---|
| `daynight_stream.py --year {2016,2019} [--exit-x --exit-y]` | streams interactions, splits day (09–18) / night (21–06) in Berlin time, produces the age-vs-distance day/night comparison |
| `spatial_map.py --year --side` | full-comb maps of mean phase / age / density (used to see where the entrance is) |
| `entrance_optimize.py --year --side` | grid-searches the entrance point that best explains the phase gradient (validates the method on colony B) |
| `full_seam.py` | full-data x/y histograms + 2-D density to check the 2016 two-camera stitch |

## data/
- `dist_exit_df_side0_2016_CORRECTED.csv` — position-aggregated cosinor estimates for colony A, distances from the **corrected** entrance `(346,270)`. Feeds Note 2 + Fig S1e without needing the 15 GB pickle.
- `dist_exit_df_side0_2019_recomputed.csv` — same for colony B (entrance `(5,264)`, unchanged).
- `daynight_2019.csv`, `daynight_2016_fixed.csv` — per-distance-bin day/night median ages.

## figures/
- `spatialmap_{2016,2019}_side0.pdf` — phase/age/density maps (the entrance evidence).
- `full_seam.pdf` — the 2016 stitch check.
- `bg_2019_cam0_side0.png`, `bg_2019_cam1_side1.png` — median-background frames showing the physical entrance/tube (colony B, both sides).
- `bg_2016_sample1.png` — a 2016 calibration frame (checkerboards mark the multi-camera overlap).
- `daynight_2019.pdf`, `daynight_2016_fixed.pdf` — day/night age-vs-distance.

---

## Reproduction environment (how the pipeline was actually run)

The interaction data is `beesbook`-owned on trove and does not fit in RAM on the
login/analysis nodes, so the real `aggregate_…per_hive_position.py` was run on
**horus** (251 GB RAM), reading the data via **thekla** (which mounts
`/mnt/trove` as `beesbook`):

```bash
# copy a pickle from trove (via thekla) onto horus, then run the real pipeline
ssh horus 'ssh thekla "cat /mnt/trove/storage/beesbook/speedtransfer/data/zenodo/2016/interactions_side0.pkl"' \
    > ~/circadian_data/interactions_side0_2016.pkl
# bb_rhythm's filter_overlap / combine_bees are pure pandas; the module's cv2 /
# bb_behavior imports were satisfied with empty stubs on PYTHONPATH.
SPEEDTRANSFER_DATA_PATH=~/circadian_data \
PYTHONPATH=~/pystubs:~/bb_rhythm_src:<repo> \
  python analysis/aggregate_cosinor_estimates_per_hive_position.py --year 2016 --side 0
```

`bb_rhythm`: https://github.com/BioroboticsLab/bb_rhythm (only `interactions.filter_overlap`
and `interactions.combine_bees_from_interaction_df_to_be_all_focal` are used).
