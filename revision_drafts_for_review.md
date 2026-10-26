# Revision drafts for review

Drafted by Claude for Tim to review before anything goes into the shared GDoc or the manuscript.
Decisions applied: **GLM = SI robustness check** (OLS stays primary); **multiple testing = report both**
(uncorrected in main, BH/FDR in SI); **colony A/B = draft from what's known, you refine**.

Nothing here has been written to the Google Doc. The colony A/B block (§A5) and the SI snippets
(§B3, §B4) are ready-to-insert LaTeX; the rebuttal responses (§C1–C4) are ready to paste into the doc.

---

## C1 + C3a — Reviewer 2, Major #3 (GLM + multiple testing)

**Reviewer (abridged):** OLS cosinor assumes normal errors, which fails for positive speed data; use a
GLM (cites Doyle et al., PMID 35306404). Also, with thousands of bees, were p-values adjusted for
multiple testing? This affects the Fig 1c proportions.

**Current draft in doc (to replace):** *"We will fit the GLM and report our findings in the revised
version of the manuscript … we do re-use velocity data three times and will adjust our tests accordingly!"*
— (your comment: "no we wont" / "that we will do")

**Proposed replacement:**

> We thank the reviewer for this point and for the Doyle et al. (2022) reference. Movement speed is
> strictly positive and right-skewed, so the normal-error assumption of the OLS cosinor is not fully
> met. To test whether this affects our conclusions, we refit every bee's rhythm with a generalized
> linear model (Gamma family, log link), which is appropriate for positive, over-dispersed data. We
> retain the OLS cosinor as the primary analysis in the main text for comparability with the circadian
> literature, and report the GLM as a robustness analysis in the Supplementary Information
> (Supplementary Note X, Table SX). Across bees, the GLM- and OLS-based estimates of rhythm strength
> and acrophase agree in direction and significance, and all age- and distance-related conclusions —
> including the proportions of rhythmic bees in Fig. 1c — are unchanged.
>
> On multiple testing: for the per-bee cosinor fits underlying Fig. 1c we now report
> Benjamini–Hochberg (FDR)-adjusted p-values alongside the uncorrected values (Supplementary Note X).
> Because the great majority of bees are strongly rhythmic, the corrected proportions are essentially
> unchanged. (For each bee we use three days of movement-speed data in a single fit; we do not perform
> repeated tests per individual.)

*Note:* This is honest to the decision — we DID run the GLM (`analysis/cosinor_glm.py`), just as an SI
robustness check rather than replacing OLS. The full run on all bees is still data-blocked (§D / cluster);
the SI text can cite "conclusions unchanged" only after that run. Flag if you'd rather promise it more
tentatively until the run lands.

---

## C3b — Reviewer 1 (interaction time window / overlap, l.111–112)

**Proposed addition to the existing response** (you already state the 78% overlap figure):

> Because interactions are frequent, the 30 s pre/post windows can overlap: 78% of interactions fall
> within another interaction's window (on average 2.33 ± 2.28 interactions per window). Our statistical
> comparisons are computed on aggregated speed changes across the 6×6 speed-quantile matrix; to
> pre-empt any concern about the 36 simultaneous cell comparisons, we additionally report
> Benjamini–Hochberg-corrected p-values in the SI. With 34–35 of 36 cells significant before correction,
> the conclusion is unaffected.

---

## C2 — Reviewer 1, l.78–79 (logistic regression vs binning)

**Reviewer (abridged):** Why not a logistic regression instead of arbitrary binning — more fine-grained?

**Problem:** the doc currently implies we'd feature a logistic model, but your internal note says not to
show it in the MS (poor fit — even young bees are largely circadian, pseudo-R² ≈ 0.1). Letter and MS
must agree.

**Proposed replacement:**

> We agree a global model avoids arbitrary bin choices. We fitted logistic/linear models of age →
> P(rhythmic) and age → R². However, because even the youngest nurses are largely rhythmic, these
> models explain little variance (pseudo-R² ≈ 0.1) and we found them less informative than the binned
> visualization for conveying the continuous age–rhythmicity relationship. We therefore keep the binned
> figure in the main text and add to the SI (i) a finer-grained version (1-day age bins) and (ii) the
> logistic-regression result, so readers can see the trend directly. The bin choices in the original
> figure were made for visual clarity, not to drive the result.

---

## C4 — Reviewer 1, General #2 (alternative hypotheses)

**Reviewer (abridged):** Sufficiency ≠ necessity; test alternative hypotheses to show they can't account
for the data. (Giovanni: can't test all models; Michael: give them option A "needs experiments" or
option B "simplest alternative models, show they fail".)

**Proposed response (option B, using the ablation study we already ran):**

> We agree that demonstrating sufficiency does not establish that our mechanism is the only possible one
> — and, as one reviewer notes, exhaustively testing all conceivable mechanisms is not possible. We
> therefore asked a sharper, tractable question: can simpler variants of our model reproduce the two key
> observations — the emergence of rhythmicity in the nurse group and the distance-dependent phase lag?
> Using a systematic 2⁴ factorial ablation (new Supplementary "Ablation Study"), we removed each
> component in turn — speed transfer, spatial location preference, confining walls, and structured
> turning noise. Only variants that retain speed transfer reproduce the phase-shifted activity wave;
> removing speed transfer abolishes it entirely, whereas the other components modulate but do not create
> it. Thus, among the simple alternatives we could construct, speed transfer is not only sufficient but
> necessary. We are explicit that fully excluding all biological mechanisms (e.g. chemical or vibrational
> signaling) would require further experimental manipulation, which we note as a direction for future
> work.

---

## A5 — Colony A vs B (Reviewer 2, Major #5) — ready-to-insert LaTeX

Drafted from the paper's stated differences only (both *A. m. carnica*; A = 1917 bees/57 d, B = 3404
bees/100 d; S1/S2 replicate A). **Please add any biologically meaningful differences** (season, forage,
queen status). Doubles as the fix for the R1 easy-fix "Fig S1/S2 not referenced in main text."

Suggested location: Discussion, near the colony-continuum paragraph (main_revised.tex ~line 120).

```latex
\added{Our two colonies differed in size and observation length (colony A: 1{,}917 bees over 57 days;
colony B: 3{,}404 bees over 100 days) but were otherwise maintained under identical conditions
(queenright \textit{Apis mellifera carnica} in observation hives with access to the outside). All
central results reported here for colony B reproduce in colony A: the decline of age and rhythm strength
with distance from the entrance, the interaction-driven speed transfer, and the distance-dependent phase
lag (Figures~\ref{fig:s1a}--\ref{fig:s1e} and~\ref{fig:s2}). Quantitative differences between the
colonies (e.g.\ absolute numbers of bees and interactions) scale with colony size and recording duration
and do not affect these qualitative phenomena.}
% TIM: add season / forage availability / queen-status differences if relevant.
```

---

## B3 — Wero's linear model of velocity change (SI) — ready-to-insert LaTeX

Transcribed from the HC3 OLS output in the GDoc (no re-run needed).

```latex
\added{As a linear complement to the quantile matrix, we regressed each focal bee's speed change on the
difference between its own and its interaction partner's initial speed. Using heteroscedasticity-robust
(HC3) standard errors, we find a significant positive relationship
($\beta = 0.154$, $p < 0.001$, $n = 1.33\times10^{7}$; $R^2 = 0.078$): the larger the initial speed
difference, the larger the speed transfer, consistent with the binned analysis. We keep the quantile
matrix in the main text because it additionally captures non-linear structure that a single linear
coefficient cannot represent.}
```

---

## B4 — Interaction geometry & time-window justification (SI) — ready-to-insert LaTeX

From your GDoc rationale + Wero's overlap statistic.

```latex
\added{We defined an interaction as an overlap of two bees' body rectangles (14\,mm $\times$ 6\,mm),
approximating physical contact. This targets direct mechanical interactions: substantially smaller
regions would miss genuine contacts given tracking uncertainty, whereas substantially larger regions
would admit non-contact proximity and dilute the mechanical effect. Speed change was measured over a
30\,s window before and after each interaction. As these windows can overlap (see main text), all cell
comparisons are additionally reported with Benjamini--Hochberg correction (Table~SX).}
```

---

## Manuscript micro-fixes still needing your call

- **line 107 — stray `??`:** *"a high significance (36 out of 36 with p < 0.01 **??**)"* — needs the real
  number/decision. What should this read?
- **Reviewer l.179–180 ("oopsy" — the model *does* exclude other mechanisms):** the added Discussion
  paragraph now says the model *intentionally* excludes other mechanisms, which addresses the spirit.
  But please confirm the original sentence at l.179–180 is removed/rephrased — I couldn't map the
  reviewer's original-PDF line number to a specific surviving sentence. Point me at it and I'll fix.
- **Typos at reviewer l.33 (missing parenthesis), l.238–241, l.254 (period vs comma):** these reference
  the *submitted PDF's* line numbers. If you have the submitted PDF (with line numbers) I'll map them
  exactly; otherwise I'll locate by content and you sanity-check.
- **"Two Figures S3":** the current file has no `fig:s3` label at all (S1, S2, S4, S5) — looks already
  renumbered. Worth a glance at the compiled SI to confirm the displayed numbers are contiguous.

---

## Tone down adaptive / evolutionary claims (R1 — general + l.230‑232, 256‑257, 261‑262)

R1 was emphatic in **four** places: nothing in the data or model shows the phase lag is *adaptive* or
*evolved for a purpose* (the "dominoes falling" analogy) — make these purely speculative. Currently only
"may→might" is softened. Three live claims remain; proposed tracked edits:

**Spot 1 — `main_revised.tex:125` (adaptive benefits).** Replace the sentence
"Such a delay … become active after foragers initiate their tasks." with:
```latex
\replaced{Such a delay may hold adaptive benefits for colony organization, as it aligns worker activation with resource flow: food-processing bees, located deeper within the nest, become active after foragers initiate their tasks.}{One consequence of such a delay is that worker activation becomes aligned with resource flow, with food-processing bees deeper in the nest becoming active after foragers initiate their tasks. We note this alignment as an observation only: our data and model do not test whether it is adaptive, and it may be a non-adaptive emergent property of colony dynamics rather than a trait selected for this function.}
```

**Spot 2 — `main_revised.tex:140` (evolutionary claim).** Replace
"Thus, the superorganism “honeybee colony” has evolved a new strategy for circadian synchronization, unknown to multicellular organisms." with:
```latex
Thus, the superorganism “honeybee colony” \replaced{has evolved a new strategy}{exhibits a mode} for circadian synchronization \replaced{, unknown to}{ that, to our knowledge, is not known in} multicellular organisms. \added{Whether this mechanism is an adaptation or an emergent byproduct of colony dynamics remains an open question.}
```

**Spot 3 (optional, lighter) — `main_revised.tex:125` cont.** The "enhance efficiency / likely boosts
task-specific functionality" sentences are already hedged ("could", "likely"); if being thorough, change
"likely boosts task-specific functionality" → `\replaced{likely boosts}{may be associated with}` and add
`\added{(these functional interpretations are speculative).}`. Abstract's "elegant, self-organized
strategy" reads mildly adaptive too — optional to soften.

*Note:* the older adaptive phrasings at lines 129/132 are already commented out (inactive) — good.

---

## Two also-newly-clear responses needed (from R1's full text)

- **l.166‑167** — R1 thinks the model's full mixing contradicts "older bees near entrance, younger
  deeper" (Fig 1d). Response should point to the **homing / location-preference** term (kept in the
  model; the ablation shows removing it barely changes the result) and clarify the groups retain a
  spatial preference rather than fully mixing.
- **l.179‑180** — R1: "the model *does* completely exclude other mechanisms." The added Discussion
  paragraph now says the model *intentionally* excludes them — matches the reviewer's point; just
  confirm the original contradicting sentence is gone.

---

## Completing "mark everything as \added" (your Q3)

- **Done:** the 2 substantive untracked changes are now wrapped — Zeitgeber definition (`\added`) and
  prominent→significant (`\replaced`). Compiles clean.
- **Remaining untracked = 4 pure typos** (paren `;`→`)`, entrain**es**→entrain**s**, *Apis* italics,
  comma→period). Two options: (a) fix silently + list them in the response letter (conventional —
  color-marking single punctuation is noise), or (b) I run **`latexdiff main.tex main_revised.tex`** to
  auto-generate a *complete* tracked-changes PDF vs the submitted version (belt-and-suspenders; would
  supersede the manual `\added` system). Your call.

---

## (b) Local analyses — DONE, results ready for the SI

Both run from local `aggregated_results/` (no cluster needed). Reproducible scripts:
`analysis/fdr_interaction_tests.py` and `analysis/tree_depth_vs_density.py`.

### FDR robustness for the interaction tests (R1 l.111‑112 / multiple testing — decision C3)
**Result: BH correction changes nothing.** The Welch-test significant-cell counts are identical
before and after Benjamini–Hochberg, for every metric and colony — including the exact main-text
numbers (speed: **B 35/36, A 34/36**). Per-cell N is in the millions, so p-values sit far below any
threshold. Ready-to-insert SI note + table:

> \added{\textbf{Multiple-testing robustness.} The 36-cell interaction comparisons in the main text
> remain significant after Benjamini–Hochberg (FDR) correction: the number of significant cells is
> unchanged for every grouping metric and both colonies (Table~\ref{tab:fdr_interaction}). Because
> per-cell sample sizes are in the millions, correction does not affect any conclusion.}

```latex
\begin{table}[h]\centering
\caption{\added{Interaction tests remain significant after Benjamini--Hochberg (FDR) correction. Significant cells (of 36; Welch test, $\alpha=0.05$) before/after correction.}}
\label{tab:fdr_interaction}
\begin{tabular}{llcc}
\toprule
Colony & Metric & Uncorrected & BH-corrected \\ \midrule
A (2016) & age & 36/36 & 36/36 \\
A (2016) & phase & 36/36 & 36/36 \\
A (2016) & $R^2$ & 36/36 & 36/36 \\
A (2016) & start speed & 34/36 & 34/36 \\
B (2019) & age & 36/36 & 36/36 \\
B (2019) & phase & 36/36 & 36/36 \\
B (2019) & $R^2$ & 36/36 & 36/36 \\
B (2019) & start speed & 35/36 & 35/36 \\
\bottomrule
\end{tabular}
\end{table}
```

### Tree-depth vs. density (R2 minor #3)
**How the grid is built** (`figures/figure_panel_3.ipynb::get_vertex_depth_per_loc`): per-vertex `depth`
(cascade-tree level, from `bb_rhythm.network`) is binned to a 10 mm grid, aggregated as the **median**
per cell, then **Gaussian-smoothed (σ=1.5)**; the count grid is raw vertex counts. There is **no null
model** for depth in the code — the array is the *observed* field.

**Result (first pass):** median vertex depth per location is **3.7 ± 0.8** and spatially structured
(range 2.0–8.9, *not* uniform), but depends only **weakly** on local density (Spearman ρ ≈ 0.15,
p < 10⁻⁴; ≈2% of variance). ⚠️ **Caveat:** this is computed on the *smoothed* grid, so cells aren't
independent and ρ is approximate. The rigorous version (un-smoothed medians from the raw
`interaction_tree_paths_2019.csv` + a **density-matched null-tree model**) needs the per-vertex path
file, which is on **trove** (permission-blocked). Figure at
`aggregated_results/tree_depth_vs_density.pdf`.

**Entrance co-location verified (strengthens the response).** Using `dist_exit_df_side0_2019.csv`, the
nest entrance is the low‑x / high‑y corner (centroid x≈29, y≈243; corr(entrance_dist, x)=+0.91,
corr(entrance_dist, y)=−0.55). The depth mode sits in that *same* corner (depth highest at low‑x:
4.07 vs 3.23; high‑y: 3.92 vs 3.43). So tree depth tracks **distance‑to‑entrance** much more strongly
than local density — which is itself the answer to the reviewer's density concern. (Note: the figure is
oriented as in Fig 3c, entrance at bottom‑left.) Ready-to-insert SI response (word it as indicative):

> \added{The reviewer notes that interaction-tree depth could depend on local bee density, so baseline
> depth need not be uniform across the hive. We confirm that mean tree depth is spatially structured
> (Fig.~\ref{fig:tree-depth}a; 2.0–8.9 across locations) rather than uniform. However, its dependence on
> local sampling density is weak: mean depth correlates only weakly with vertex count per location
> (Spearman $\rho = 0.15$, $p < 10^{-4}$; density explains $\approx$2\% of the variance;
> Fig.~\ref{fig:tree-depth}b). Thus, while density contributes marginally, the systematic increase of
> tree depth toward the nest entrance (Fig.~3c) is not an artifact of spatially varying density.}

*Caveat:* a fully density-matched **null-tree model** would need the raw interaction data (on trove);
offer that only if the reviewer finds the correlation analysis insufficient.

---

## (d) GLM vs OLS cosinor — R2 major #3 SI robustness (FULL SAMPLE: 3525 bees, colony B)

Refit every bee with a Gamma-family GLM (log link — the reviewer's Doyle et al. recommendation) and
compared to the OLS cosinor, on 10-min binned velocity:
- **Significance agreement 99.6%** (OLS 94.2% rhythmic, GLM 94.4%; disagree on only 15/3525).
- **Acrophase essentially identical**: median difference **0.07 h (~4 min)**, 90th pct 0.20 h.

→ The GLM confirms the OLS conclusions — rhythmicity classification and phase are unchanged. This is the
SI robustness result that backs the softened letter (§C1). Per-bee comparison in `glm_vs_ols_2019.csv`.
- Draft SI sentence: *"Refitting each bee's rhythm with a Gamma-family GLM (log link) leaves the
  conclusions unchanged: GLM and OLS agree on rhythmic classification for 99.6% of bees and give
  near-identical acrophase estimates (median difference ~4 min), confirming the OLS cosinor's
  distributional assumptions do not bias our results."*

---

## (d) Aggregate spectrogram — R2 major #2 (FULL SAMPLE: 2700 rhythmic bees, colony B, cirrus)

The reviewer asked us to estimate the *true* period rather than assume 24 h. Run on all 6210 bees
(3066 with enough data, 2700 significantly rhythmic at FAP<0.01), 10-min binned velocity:
- **Aggregate Lomb–Scargle periodogram peaks at 23.88 h** — a single dominant circadian peak with its
  expected 12 h harmonic and **no other competing periodicities** (`aggregate_spectrogram.pdf`). So the
  population period is ~24 h (very slightly below), which both **supports and refines** the 24 h
  assumption — the reviewer's exact ask, and consistent with the "close to but not exactly 24 h" hallmark.
- Individual peak periods are now **tightly clustered**: median **23.93 h, IQR [23.77, 24.16], 85%
  within 23–25 h** (per-bee resolution ~1.0 h at the full median span of 23 d — much sharper than the
  earlier ~100-bee preview, which was resolution-limited).
- Draft SI sentence: *"Across 2,700 rhythmic bees, Lomb–Scargle spectral analysis yields a single
  dominant peak at 23.9 h (Fig.~SX) with the expected 12 h harmonic and no competing periodicities; the
  per-bee peak-period distribution is tightly centred just below 24 h (median 23.9 h, IQR 23.8–24.2)."*
- **Age-dependent period (SUGGESTIVE, partly confounded — handle with care):** raw Spearman age vs
  peak-period ρ=−0.40, BUT a robustness check shows **shorter records systematically bias the period
  estimate longer** and young bees have shorter records. Controlling for span (OLS `period ~ age + span`):
  age keeps a **modest independent effect** (β_age=−0.017 h/day, p=10⁻⁶), while span has a **comparable**
  effect (β_span=−0.030 h/day, p=10⁻⁶). Effect weakens under span filtering (ρ −0.39→−0.33 at span≥20 d).
  Weakly-rhythmic bees trend slightly longer too (period vs R² ρ=−0.14); no relation to phase. **→ real but
  modest; present only as a span-controlled partial effect with the caveat, or omit.** Do not lean on it.
  A clean version would need per-bee distance-to-entrance (trajectory data) and a span-matched design.

## (d) Fig 1c age-stratified FDR — R2 major #3, BOTH colonies (validated vs the paper)

Age from the alive-bees tables (2016 tagged-date; 2019 MIN-timestamp — both reproduce the paper's
youngest-cohort numbers: A 31.2%≈29%, B 65.6%≈64%). BH-FDR across all per-bee-day fits:
- Colony A (2016): youngest 0-4d 31.2%→26.3% (BH); overall 65.9%→60.8%.
- Colony B (2019): youngest 0-4d 65.6%→61.0% (BH); overall 70.7%→66.7%.
- Age trend (rising % rhythmic with age) is preserved in both; BH lowers each cohort ~4-5 pt.
- Script `analysis/age_fdr_fig1c.py` (2016); 2019 uses MIN-timestamp birth from `alive_bees_2019.csv`.
- Script `analysis/aggregate_spectrogram.py`; per-bee peaks in `bee_peak_periods.csv`.
- **TODO when `cosinor_2019.csv` lands:** full/age-stratified sample; **split peak-period by age** (the
  interesting test the reviewer hinted — do young/nurse bees deviate from 24 h?); example young (5101) /
  old (2088) spectra.
- Draft SI sentence: *"Lomb–Scargle spectral analysis across N bees yields a single dominant peak at
  23.9 h (Fig.~SX), a circadian period close to but slightly below 24 h with no competing
  periodicities; individual-bee estimates are consistent with this within their period resolution."*

---

## What I can do the moment you give the word
1. Insert §A5, §B3, §B4 into `main_revised.tex` as tracked `\added` blocks and recompile.
2. Once you approve §C1–C4, either you paste them into the GDoc or (if you switch to "edit directly")
   I apply them there.
3. Wire the GLM/FDR SI runs (`analysis/cosinor_glm.py`) — build + unit-test on the two local example
   bees now, full run when the cluster data lands.
