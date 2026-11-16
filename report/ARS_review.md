# ARS panel review: "Do Language Models Defend What They Claim to Know?"

**Mode:** `academic-paper-reviewer` full panel, time-boxed inline run, NOT_CALIBRATED. One model ran all five seats in one context (role-separated, not independent). The Phase 0 card pause was skipped as the task instructed. Checker scripts and advisory attachments did not run. Interim numbers are not critiqued.

**Panel:** EIC = Track 1 judge (novelty vs Kumaran et al.). R1 = evaluation statistician. R2 = sycophancy/calibration researcher. R3 = advice-taking researcher. DA = Devil's Advocate.

## 1. Decision and scores

**Decision: Minor Revision.** The design is sound, and the W2W control and ĉ50 framing are real contributions. Four issues affect the headline metrics; each can be fixed with existing data or wording. The DA's CRITICAL candidate (causal reading of confidence → resistance) was downgraded to MAJOR: the main claim is correlational and holds once hedged (item 7).

| Rubric dimension | Now | After MUST+SHOULD fixes |
|---|---|---|
| Impact & Innovation | 4 | 4 |
| Execution Quality | 3 | 4 |
| Presentation & Clarity | 4 | 4–5 |

## 2. Revision items (ranked by expected score gain)

1. **MUST: ĉ50 and the OR are pooled over turns 1–3** (§3.4, Table 4, abstract, §4.3). `extra_stats.c50()` fits on every row of `decisions()`. Turns 2–3 use confidence stated after a push, include only items that survived earlier turns, and the P2/P3 pushes claim a higher user reliability. A single "implied r" is therefore not identified. *Fix:* report turn-1-only ĉ50/OR as the headline (filter `decisions()` to `turn==1`) and keep the all-turn fit as secondary. Add one sentence explaining why. [R1, R3, DA]
2. **MUST: "kept/retained" in the paper means "did not adopt X" in the code** (§3.3, Table 4 defense gap, Table 5 keep rates). `retained_correct_under_C2W`, `defense_gap` and `keep_bins` count a switch to a *third* wrong option as keeping a correct answer, and the AUROC label "resisted" does the same. *Fix:* define kept as final answer == gold, or rename the measure "did not adopt the target". Report the third-option switch rate per arm. [R1, DA]
3. **MUST: the yardstick contradicts the W2W interpretation** (Table 1 W2W row, §4.2 ¶2, §5.1, abstract). Equation (1) says an agent with c < r should adopt *any* asserted option, so W2W adoption below the threshold is consistent with the yardstick, not "compliance unrelated to correctness". *Fix:* describe W2W as measuring how far yielding ignores which option is asserted. Describe latent discernment as private knowledge beyond the yardstick (the second-choice argument in Appendix A). Keep the point for users: a model's agreement is weak evidence that the user was right. [R3, DA]
4. **MUST: the defense gap has no normative zero** (§3.3, §4.3, §6). Under Eq. (1), a correct answer with c > r should be kept with probability 1, not with probability c. *Fix:* rename it "confidence–retention gap" (descriptive) and drop "as often as their confidence implies", or add the yardstick prediction (share of C2W items with c0 > ĉ50) and compare against it. [R3, DA]
5. **SHOULD: the paper says CIs are item-clustered, but the code does not cluster them** (§3.4 "again with item-clustered intervals"). The §4.3 target-correct β comes from `analyze.logit_boot`, which resamples decisions, not items; `boot_mean_clustered` does not cluster; the pooled gpt-oss McNemar treats one item under two models as two independent pairs. *Fix:* resample by `qid`, reusing the loop in `c50()`, or state the limitation. [R1]
6. **SHOULD: the neutral baseline is asymmetric** (Table 3 "W2C beyond neutral re-ask", §4.2). It subtracts a 1-turn neutral fix from 3-turn cumulative W2C adoption (`extra_stats.py` l.108). *Fix:* make turn-1 W2C minus the neutral fix the headline (`W2C_adopt1` is already computed) and keep the 3-turn figure as an upper bound. [R1]
7. **SHOULD: causal language about confidence** (§1 ¶3, §4.3, §5.1). Stated confidence and resistance may both track latent item knowledge. The pooled turn-1 AUROC also mixes right items and wrong items. *Fix:* add the already-computed C2W-only AUROC (`auroc_c2w`) to Table 4 and use association wording (§3 below). [DA, R2]
8. **SHOULD: the scope is wider than the evidence** (abstract, §6, Table 4). The inferential claims rest on one model family (two gpt-oss sizes, pooled). Qwen is called "descriptive only" yet gets an OR, ĉ50 and CIs. *Fix:* replace "the stronger models" with "the two gpt-oss models". Move the Qwen rows to the appendix or blank its inferential cells. Give per-model N in the abstract. [EIC, R2]
9. **SHOULD: degenerate CIs** (Tables 2–4: "0 [0, 0]", "100 [100, 100]", "0.50 [0.50, 0.50]"). Percentile bootstraps collapse at the boundaries. *Fix:* use Wilson or Clopper–Pearson intervals for single proportions and mark degenerate AUROC CIs as "n/a". [R1]
10. **COULD: exclusions and item overlap** (§3.4, §3.1). Items failing mid-ladder (`job()` except branch) or unparsed at T0 are dropped; "unrelated to item content" is untested; models answered different subsets. *Fix:* report n_total vs n_valid and failed-item categories per model; compare 120b vs 20b on shared items. [R1]
11. **COULD: sharpen novelty against Kumaran et al.** (§1 ¶2, §2). One sentence: token-probability vs verbalized confidence, advice of stated accuracy vs content-free pressure, no control vs W2W; ĉ50 is the counterpart of their advice-accuracy manipulation. Add "to our knowledge" to claims about what prior work lacks. [EIC, R2]
12. **COULD: register** (abstract, throughout). Cut the abstract to three headline numbers. Use one outcome verb ("adopt", not also yield/surrender/capitulate). [EIC]

## 3. Overclaims and hedged rewrites

- Abstract: *"Whether a model's behaviour honours this signal under challenge has not been measured directly."* → "Few studies relate verbalized confidence to behaviour under argument-free challenge."
- Abstract: *"…on 201 MMLU-Pro questions."* → "…on a stratified sample of 201 MMLU-Pro questions (each model completed [min]–[max])."
- Abstract: *"…a failure that calibration metrics alone cannot reveal."* → "…a behavioural consequence that calibration metrics do not directly measure."
- §1: *"…any systematic relation between stated confidence and yielding reflects how the model treats its own confidence, not how it weighs new information."* → "…such a relation cannot come from argument content; it may reflect stated confidence or the item knowledge that also drives it."
- §3.3: *"…discernment reflects only the model's initial correctness."* → "…discernment compares different items, so it combines target correctness with differences in item difficulty and confidence; latent discernment removes these."
- §3.4: *"These failures were transient server or rate-limit errors unrelated to item content."* → "These failures were server or rate-limit errors; we did not test whether they depend on item content."
- §4.3: *"It is therefore not an artefact of mixing arms."* → "The relation therefore does not appear to be only an artefact of mixing arms."
- §4.3: *"…these models act as if a bare user assertion were correct roughly three times in four."* → "…under Eq. (1)'s simplifying assumptions, the turn-1 threshold corresponds to treating a bare assertion as correct about three times in four."
- §4.3/§6: *"…retained correct answers about as often as their confidence implied."* → "…the share of correct answers not abandoned was close to mean stated confidence (a descriptive match)."
- Table 1: *"compliance unrelated to correctness"* → "adoption not specific to the correct answer."
