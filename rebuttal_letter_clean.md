# Response to Reviewers

Manuscript CURRENT-BIOLOGY-D-25-00716 — *Collective flow of circadian clock information in honeybee colonies*

Dear Cyrus,

Apologies for the delayed response. We needed additional time to revisit our data in order to assess several points raised by the reviewers, and with both first authors having moved on from the lab, coordinating revisions took longer than anticipated.

We greatly appreciate the reviewers' thorough and constructive feedback. Their suggestions are both helpful and attainable, and we have already started working to incorporate most of them. There are only a few recommendations - specifically, those that would require rerunning our entire data-processing pipeline - that are prohibitive under our current time constraints. We would like to discuss which of those remaining recommendations would need to be implemented to satisfy the reviewers' concerns and avoid rejection.

Below, you will find a point-by-point response: each reviewer comment is presented in bold, followed by our response in italics. To streamline the presentation, we have grouped all of the "easy fixes" (typos, formatting changes, minor clarifications) at the beginning, so we can focus on the more substantive concerns afterward. Where specific comments overlap with general comments, we have not repeated them to avoid redundancy - rather, we address them under the relevant general point.

Thank you for your continued consideration. We look forward to your feedback on our revision plan.

Sincerely,

Tim

On behalf of all authors

---

## Reviewer 1

### General comments

**Reviewer:** [...] the study equates activity with movement (as in a significant change in spatial coordinates), which is not always a full-proof indicator. Many activities inside the nest of a social insect colony do not necessarily involve movement (e.g., brood care) and may therefore not be captured in the data. While it may not be possible to capture these activities with the acquisition system used here, the authors should discuss this limitation in the manuscript, or provide evidence that movement is indeed a good proxy for general activity.

Response: We agree. We will discuss the use of movement as a proxy of activity and the limitations associated with this choice.

---

**Reviewer:** [...] the model does not test any alternative explanations for the observed phenomenon. Since it is unlikely that the authors can include new experiments in a revision of the manuscript, I recommend that they include instead models testing the alternative hypotheses to formally demonstrate that they cannot account for the observed data.

l.179-180: That's incorrect. The model does, in fact, completely exclude other mechanisms. This is why the authors can conclude that the proposed mechanism might be enough to explain the observations. This being said, it would be a more powerful result if these other mechanisms had been tested in simulations to show whether they are (or not) additional plausible explanations to the data. Models are useful to demonstrate the plausibility of a mechanism, but they are truly powerful when used to eliminate alternative explanations.

Response: We have carried out this analysis: a systematic 2^4 factorial ablation, removing each model component in turn (speed transfer, spatial location preference, confining walls, and structured turning noise). Only variants that retain speed transfer reproduce the phase-shifted activity wave; removing speed transfer abolishes it entirely, whereas the other components modulate but do not create it. Among the simple alternatives we could construct, speed transfer is thus not only sufficient but necessary. We present this ablation study in the revised Supplementary Information, and note that fully excluding all conceivable biological mechanisms would require further experimental manipulation, which we flag as a direction for future work.

---

**Reviewer:** I suggest that the authors reduce the number of adaptive claims in the manuscript and make it evident that they are purely speculative at this point.

l.230-232: There is no evidence that this is adaptive. It's a consequence of the dynamics of the system, but it does not mean it was evolved for that purpose.

l.256-257: Again, the observations do not substantiate this evolutionary claim. The synchronization could be a simple emergent property resulting from interaction processes at work in bee colonies that may have evolved for completely other reasons.

l.261-262: The study does not provide any evidence that the time-shift is beneficial to the colony. It shows it exists, it provides a mechanistic explanation for it, but nothing that is reported suggests that it improves the efficiency of the colony.

Response: Agreed. We will tone down the claims of the adaptive value of the phase shift or express this as speculative.

---

### Specific comments

#### Easy fixes

We agree to all of them and will correct or add contents accordingly.

- l.8: Define Zeitgeber for non-specialists.
- l.33: Typo (missing parenthesis).
- l.46-49: Vibrations are mentioned twice, first an abiotic Zeitgeber, then as a social one. I imagine the authors are not referring to the same types of vibrations in each case, but this should be clarified in the text.
- l.59: entraines -> entrains
- l.64: Species name should be italicized.
- l.72: Figure 1b instead?
- l.74: Figure 1e instead?
- l.128-130: I would caution the authors against using unnecessary physical analogies when talking about social behaviors. Momentum transfer between particles suggests conservation of energy, which is not the case in the "speed transfer" observed here. The explanation of the speed transfer is clear enough already and doesn't require a potentially misleading analogy to physical processes.
- l.183-185: The writing is a bit confusing here. At first, I thought this was done on the data generated by the model, but only understood later when reading the methods that this was actually applied to the real data.
- l.179-180: That's incorrect. The model does, in fact, completely exclude other mechanisms.
- l.43-45: It's not clear to me how the second question derives from the observations mentioned above. It's important to clarify this point because the last paragraph of the introduction emphasizes that the social angle is the one that's really investigated in this study, yet it is not clear until that point that it is.
- l.238-241: Typo. These two sentences should not have been separated by a period.
- l.254: Typo. The comma before "Accordingly" should be a period.
- Figure S1 and S2 are not referenced in the main text.
- And there are two Figures S3.

---

#### Point-by-point responses

**Reviewer:** l.54: Prominent and weak seem antithetical.

Response: We wanted to express high cosinor R² (a good fit to the sinusoidal model) but low amplitude. We agree that "prominent" is misleading and would change it to "significant". As asked by reviewer #2, we will clarify the two different (but correlated) metrics amplitude and R² in the manuscript.

---

**Reviewer:** l.78-79: Why not use a logistic regression instead? It would avoid the arbitrary binning and would offer a more fine-grained analysis of the change in rhythmicity with age.

Response: We chose binning because a "global" model would suffer from a strong bias (large numbers of young bees, much fewer old ones) and we would have to balance the data, possibly using (samples from) age bins as well. The bin choices in the manuscript were arbitrary and optimized for visual clarity. We can provide a plot with a more fine-grained binning (1d per bin) in the SI.

We fitted a logistic regression (age → "is circadian" or circadianness [R²]) but this results in a R² of 0.1, because even the youngest bees are widely circadian. We think adding this to the manuscript would not help the reader understand this relationship; we however could add the figure with finer age binning and the logistic regression to the supplementary information.

---

**Reviewer:** l.86-87: This could be easily tested statistically with a linear model.

Response: We agree, and have added this test to the revised Supplementary Information (Supplementary Note 2). An ordinary least-squares regression confirms that age significantly predicts distance from the entrance, consistently in both colonies (colony A: β = −7.23, R² = 0.19, n = 82,124; colony B: β = −6.04, R² = 0.09, n = 87,557; both p < 0.001): older bees are located closer to the entrance in both colonies. The full regression table is given in Supplementary Note 2.

---

**Reviewer:** l.88-89: This correlation should be tested statistically.

Response: We agree, and have added this test to Supplementary Note 2. Circadian phase decreases significantly with age in both colonies (colony A: β = −0.024 h per day of age, R² = 0.02; colony B: β = −0.024 h per day, R² = 0.04; both p < 0.001): older bees reach their activity peak earlier in the day, as claimed. The modest R² reflects the large scatter around a robust mean trend.

---

**Reviewer:** l.90-92: Again, this should be tested statistically.

Response: We agree, and have added these tests to Supplementary Note 2. Ordinary least-squares regressions confirm every spatial trend of Fig. 1e (all p < 0.001, in both colonies): with increasing distance from the entrance, activity phase becomes later (colony A: β = +20.8; colony B: β = +106.5 comb-units per hour of phase) and rhythm strength decreases (colony A: β = −590; colony B: β = −759 comb-units per unit R²); bees further inside the nest are correspondingly less synchronized. The full regression tables are provided in Supplementary Note 2.

---

**Reviewer:** l.109-110: Is there a justification for these dimensions? Also, how sensitive are the results to changing these dimensions?

Response: Taking into account the length and width of a bee (we here assumed 12 mm x 4 mm), we defined a 1 mm interaction region around the body which results in 14 mm x 6 mm. We have not analyzed how sensitive our results are to this definition; however, making this region smaller will result in a sharp decline in number of interactions, and increasing the size of the window will include interactions between bees that do not physically touch — both changes effectively will result in higher noise levels. We agree that it would be interesting to see how stable the results are over the size of the interaction regions, but since this is a very early parameter in the processing pipeline it would mean running all analyses again several times, which we likely will not be able to do.

---

**Reviewer:** l.111-112: Is there a justification for this time window? Also, how sensitive are the results to changing the time window? And what about cases of multiple interactions during the time window?

Response: Studying the effect of the window size on the results would be very interesting, we agree! We expect that shorter windows will effectively introduce more noise (because movement speeds are inherently noisy) and longer windows possibly average out the effect we would like to measure. However, similar to our previous response, this analysis would represent a substantial time investment we likely will not be able to realize. We found 78% of interactions to occur within the time window of another interaction. On average, there are 2.33 ± 2.28 interactions within the time window before and after an interaction. Accordingly, we now report Benjamini-Hochberg (FDR)-corrected p-values for these interaction tests in the revised Supplementary Information; all cells remain significant after correction (Supplementary Note 3).

---

**Reviewer:** l.114-115: Why use the quantiles of the data? Couldn't this be analyzed directly using a linear model approach? If the authors choose to keep analyzing the data as quantiles, then they should perform a sensitivity analysis of the results to the chosen quantiles.

Response: We previously fit a linear model but pivoted to quantiles in order to capture the non-linear effect; we now also report the linear model in Supplementary Note 2. An ordinary least-squares fit confirms that the speed difference between interaction partners significantly predicts the focal bee's speed change (β = 0.154, R² = 0.078, n = 13,289,384 interaction events, p < 0.001, with heteroscedasticity-robust HC3 standard errors): the larger the speed gap, the larger the transfer. The modest R² is expected — a single linear term cannot capture the saturating, sign-changing structure that motivated our quantile analysis.

---

**Reviewer:** l.163-164: Do fast bees lose speed from interactions as well?

Response: Yes, this is indeed the case — see figure 2a: the fastest group (6) exhibits negative speed changes after interactions with almost all speed groups.

---

**Reviewer:** l.166-167: Is that what really happens in a beehive? I thought that the older bees tended to prefer staying closer to the entrance, and the younger ones further away, as per figure 1d? If I understand correctly, the model leads to a full mixing of the two groups over time, which seems to contradict the observations.

Response: You are right. The model contains a term for each group's spatial fidelity as well. Both the rhythmic and the non-rhythmic agents are pulled back to their respective preferred locations (two points in the plane) — see methods section, line 387. Due to this bias, the two groups are not fully mixed but the degree of mutual penetration is proportional to the motion speeds. We will update the wording in the manuscript accordingly and, as part of the ablation study, we tested the effect of the location-preference (homing) term: removing it leaves the phase shift essentially unchanged, confirming it is not the driver.

---

**Reviewer:** l.382-383: For future reference, a gamma distribution is a more natural choice for modeling speed distributions than a clipped Gaussian one.

Response: Agreed. We will change that in the simulation rework.

---

**Reviewer:** l.386-388: For orientation changes, a von Mises distribution is more appropriate.

Response: Agreed. We will change that in the simulation rework.

---

## Reviewer 2

### Major comments

**Reviewer:** On page 3, line 41. The authors ask "how strict is the lack of circadian rhythm ...". I think it is important to define clearly what is meant by "circadian rhythm". I suppose the authors mean circadian overt behavior. It could be that nurse bees have molecular circadian rhythms but no overt behavioral rhythm. Although nuanced, the authors should clearly define this upfront. Also, what is the basis of using speed of movement as the metric that should be circadian? In mice, we sometimes using activity bins where we define if the mouse is active or not (as boolean).

(Reviewer #1 raised a similar point.)

Response: Agreed. We will clarify the focus on behavioral rhythmicity as quantified by movement speed and discuss the consequences of this choice.

---

**Reviewer:** The authors fit rhythm profiles assuming a rhythm period of 24h. This is a simplifying assumption one makes for either (known to be) entrained rhythms or short datasets where we cannot estimate the exact period. If indeed rhythms are entrained socially (through bumping), then it might be that the rhythm in nurse bees deviates 24h (more on why this in relevant below). Recall, that one of the hallmarks of the circadian clock is a period close to but not exactly 24h. In short, the authors have enough data to use spectrogram methods to estimate the true period of the observed rhythms (these themselves might be biologically indicative).

Response: We agree, this would be interesting! We chose to focus on the 24-hour period for this paper because for most bees this is the strongest rhythm in their movement data. Other rhythms do exist and we agree, it would be interesting to look at whether the speed transfers affect those rhythms too. We now provide a spectral (Lomb-Scargle) analysis in the revised SI (Supplementary Note 4): averaged across 2,700 rhythmic bees it shows a single dominant peak at 23.9 h with the expected 12 h harmonic and no competing periodicities, confirming a period close to but slightly below 24 h. We also examined whether the period varies with age; a weak age-dependence is present but is partly confounded by record length, so we report it cautiously.

---

**Reviewer:** The cosinor model is fit to the speed data of individual bees. The standard linear regression based cosinor fitting assumes that the errors are normally distributed. This does not hold for variables (like speed) constrained to be positive — the speed drops close to zero during the night for both bees in Fig 1b. A better approach is to use a GLM based model for such data as recommended by Doyle et al. (PMID: 35306404). Moreover, when such rhythms are fit for thousands of bees, it unclear from the paper if the p-values were adjusted for multiple testing. This would also affect the proportions provided in Fig 1c.

Response: Thanks for this comment. We have refit every bee's rhythm with a generalized linear model (Gamma family with a log link, as recommended by Doyle et al.) as a robustness check. It confirms the OLS cosinor used in the main text: the two agree on rhythmic classification for 99.6% of bees and give near-identical peak-time (acrophase) estimates (median difference about 4 minutes). We report this comparison in the revised Supplementary Information (Supplementary Note 3) and retain the OLS cosinor as the primary analysis, for comparability with the circadian literature. We do not test individuals repeatedly but indeed, for a given individual we use three days worth of movement speed data for the cosinor fit. Hence, we do re-use velocity data three times and have additionally applied a Benjamini-Hochberg (FDR) correction to the per-bee p-values. The proportion of significantly rhythmic bees decreases only modestly and the age trend in Fig. 1c is preserved in both colonies (Supplementary Note 3), leaving the conclusions unchanged.

---

**Reviewer:** The rhythms also do not look very sinusoidal. The R^2 and p-value only tests if a sinusoidal fit (with a single fundamental period) is appropriate. A strongly rhythmic non-sinusoidal pattern can still provide a lower R^2 although a rhythm still exists. This is easily rectified by considering addition harmonic terms in the cosinor regression or by precluding that there are other strong peaks in the spectrogram. Furthermore, the strength of the rhythm can be measured using the amplitude estimate or the R^2. The former measures how large the variations are and latter measures how good a fit a sine curve is. They are related but slightly different metrics. I think the authors use the latter but dont say anything about the former. Are these two consistent in the insights they provide?

Response: You are right, we use only one of the metrics because both R² and amplitude are highly correlated. For most bees, the R² indicates a good fit of speeds to the sine wave. While other rhythms exist, the 24-hour rhythm is the strongest. We now include an aggregate Lomb-Scargle spectral analysis across all rhythmic bees in the SI (Supplementary Note 4). Example spectra for a young bee (ID 5101) and an old bee (ID 2088): for bee 5101 (Fig. 1b), max_power = 0.04, 24h_power = 0.039, max_frequency = 23.686 h; for bee 2088 (Fig. 1b), max_power = 0.248, 24h_power = 0.234, max_frequency = 25.348 h. Averaging the speeds for three age groups yields a clear signal at the 24-hour period in the Fourier spectrum.

---

**Reviewer:** The authors present colony B (the longer recorded one) results in the main text and colony A in the supplements. It would be very interesting to known what the key differences or similarities were between them and if there were other experiment differences between the two recordings (e.g., season). They authors do provide the results but it is difficult for the reader to compare complex plots.

Response: We will provide a more thorough explanation of the differences of the colonies.

---

**Reviewer:** The key proposition of the study is that propagation waves of speed changes underlie the 2h difference in activity peak between the nurse and forager bees. In Page 4, line 105 "we might also expect ....., but we would expect the rhythm to be synchronized". This is not however true. Phase difference between interacting oscillators can arise due to differences in intrinsic oscillator period (hence my question above to analysis of period) or differences in strength of the driving Zeitgeber (see Figure 1c in 10.1098/rsif.2015.0282). In other words, if the externally Zeitgeber (light) entrained forager bees drive the weakly rhythmic nurse bees, all things being equal we would expect an appropriate phase difference between them depending on the strength of coupling. The authors should discuss their observations also in the context of the theory of coupled oscillators (see the above paper for starting point). This could potentially be included in the agent-based model.

Response: This is an interesting thought. Young bees would have to have a weaker coupling to light (or other abiotic factors). We are not aware of studies that would support or reject this hypothesis. Thus we would discuss this possibility in our manuscript after careful literature search.

---

**Reviewer:** Could the authors also make clear in the introduction and abstract wherein concretely lies the novelty in the paper? I understand, after some researching, that systems to track bees has been previously published by the authors and others but application to answer this particular circadian question is new.

Response: The novelty lies in our ability to track thousands of bees in the colony — a setting that previous research on circadian rhythms did not do. We will clarify this in the manuscript.

---

### Minor comments

**Reviewer:** The authors should provide the mathematical equations behind the agent-based model and the provide the parameter values used in the same. It is currently difficult to understand how the model actually works. Moreover, how many different agents were simulated?

Response: We will rework the simulation to integrate feedback of reviewer #1 and then update the methods section with equations as asked.

---

**Reviewer:** Does the distribution of the age across distance from the entrance change from day to night. If so how? How do the authors explain the increase in the number of older bees at the opposite end of the hive in Figure 1d?

Response: We have now examined this directly. Because Fig. 1d/1e are built from the start positions of recorded interactions, we split exactly that data by time of day (day 09:00–18:00 vs. night 21:00–06:00; 131 million interaction observations for colony B, the colony of the main text) and compared the age–distance profile. The result — added as Supplementary Note 5, with a new figure — is that the overall spatial age organization is stable between day and night: the young brood-nest core sits at the same distance around the clock, while the older-bee zone near the entrance extends further into the nest during the day (for example, the median age at 136 comb-units from the entrance is 18 days by day but 13 at night; a day×night interaction term is overwhelmingly significant, p effectively zero). This is expected, since the older, strongly rhythmic bees near the entrance are most active during the day and dominate daytime interactions, whereas at night the interacting population is more uniformly young. In short, the spatial age structure itself is stable; only its daytime amplitude is enhanced by the activity of the older bees.

The increase in the number of old bees is due to the geometry of the hive and the nest. Note that due to the exit being in the lower corner of the rectangular frame, the largest possible distances correspond to hive locations in the opposite corner of the hive. The brood nest, located in the center of the frame, is densely packed with younger bees so if older bees want to reach the honey storage in the topmost nest region, they often go around the brood nest, instead of through it, leading to those bees covering the top corners too.

---

**Reviewer:** Does not the interaction tree depth depend on the density of bees around a bee and so baseline depth would not be uniform across the hive? How does one control for this?

Response: Yes, in general, the density of bees directly translates to the number of potential interaction partners and, hence, the chance of being bumped. On the other hand, in dense regions bees might not be able to move as quickly, which might reduce the speed transfer effects. For our analysis we focus on young bees (< 5 days of age, predominantly in the brood nest), select speed-increasing interactions only, and limit the duration of cascades to two hours maximum. This means that all chains of speed-increasing interactions will end in roughly the same hive region, i.e. where those young bees are — the brood nest. If density creates a bias for interactions, we would see longer chains in the brood nest and, since we limit their duration, we would localize the start of these chains there too. However, our results show that the longest chains start in the brood nest (see figure 3c).
