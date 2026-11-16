# Final pre-submission check (18:2x MYT, 16 Nov 2026)

The re-review ran inline with `academic-paper-reviewer` (re-review mode). It is NOT_CALIBRATED and checks only the PDF text against the JSON.

## 1. Compliance and consistency (C)

| Check | Result | Note |
|---|---|---|
| ≤8 pages before refs/appendices | PASS | Body ends p. 8 |
| Abstract 150–250 words | PASS | 245 |
| Author + affiliation | PASS | |
| Limitations and Dual-Use / Ethical appendix | PASS | App. D |
| LLM usage statement | PASS | p. 12 |
| Repo public, README, PDF link | PASS | |
| SUBMISSION.md title and summary = PDF | PASS | Identical |
| Abstract/summary numbers | **FAIL** | "43–78" is wrong: ALLaM n_valid = 129 (fix 1); others match |
| Tables 2–5, §4, slides vs JSON | PASS | Every number checked |
| `[[`, NaN, U+FFFD | PASS | None |
| Project image | FAIL (minor) | Title clipped (fix 11) |
| SUBMISSION.md email | NOTE | Says "your email": type it into the form |

## 2. Re-review of ARS items

| # | Item | Status |
|---|---|---|
| 1 | Turn-1 ĉ50/OR headline | Resolved |
| 2 | Strict "kept"; third-option rates | Resolved |
| 3 | W2W vs yardstick | Resolved |
| 4 | Confidence–retention gap + yardstick 97% | Resolved |
| 5 | Item-clustered CIs | Partially: §3.4 says 2,000 resamples, but `c50()`/`logit_tc()` use B=500 (fix 4) |
| 6 | Turn-1 W2C minus neutral | Resolved |
| 7 | Causal language; C2W-only AUROC | Partially: AUROC added; causal wording left in the abstract, §5.1 and §6 (fix 5) |
| 8 | Scope to gpt-oss; per-model N | Partially: scope done; the abstract N range is wrong (fix 1); Table 3 still gives Qwen p-values (acceptable, caption says descriptive) |
| 9 | Degenerate CIs | Resolved (Wilson; "–" AUROCs) |
| 10 | Exclusions | Partially: API failures reported; T0-unparsed drops (Qwen 46→43, ALLaM 149→129) are not explained (fix 3) |
| 11 | Novelty vs Kumaran | Resolved |
| 12 | Register | Partially (COULD): five numbers; mixed yield/surrender/capitulate. No fix needed |

Overclaim rewrites: 9 of 10 applied. The "201 MMLU-Pro" rewrite has the wrong maximum (fix 1). The slides still carry two pre-review overclaims (fixes 2, 6). **Verdict: Accept after the MUST fixes.**

## 3. Citations (B)

- All 14 references are cited, and every citation has a reference.
- The 12 arXiv IDs match the arXiv API on title, authors, year and venue.
- [12] matches the sprint reading list. [14] (OpenAI blog) was not checked by machine.
- No metadata errors.

## 4. Fixes

After editing `report.html`/`slides.html`, re-render, recopy to `Submission/`, and confirm that the body still ends on p. 8.

**MUST**

1. `report.html` abstract: `([[N:qwen3.8-27b:n_valid]]–[[N:gpt-oss-120b:n_valid]] items completed per model)` → `([[N:qwen3.8-27b:n_valid]]–[[N:allam-2-7b:n_valid]] valid items per model)`
   `SUBMISSION.md` summary: `(43–78 items completed per model)` → `(43–129 valid items per model)` (word count unchanged)
2. `slides.html` l.40: `Nobody checks whether stated confidence predicts what a model will <b>defend</b>.` → `Few studies check whether stated confidence predicts what a model will <b>defend</b> (closest: Kumaran et al., 2025).`

**SHOULD**

3. `report.html` §3.4: `we did not test whether they depend on item content. Unparseable answers` → `we did not test whether they depend on item content. Items without a parseable T0 answer and confidence were also excluded (qwen3.8-27b: 3, allam-2-7b: 20). Unparseable answers`
4. `report.html` §3.4: `item-clustered bootstrap (2,000 resamples)` → `item-clustered bootstrap (2,000 resamples; 500 for the logistic threshold and regression models)`
5. Causal wording:
   a. `report.html` abstract **and** `SUBMISSION.md`: `but their overconfidence shields errors from correction.` → `but their overconfident errors were harder to correct.` (246 words)
   b. `report.html` §5.1: `A model that honours an inflated confidence resists valid correction, so miscalibration becomes a failure to accept correction as well as an inaccurate number.` → `Wrong answers stated with high confidence were corrected less often, so miscalibration shows up as reduced acceptance of correction as well as an inaccurate number.`
   c. `report.html` §6: `the same mechanism shields their errors from valid correction.` → `their errors stated with high confidence were less often corrected.`
6. `slides.html` l.63: `they behave as if a bare assertion is right about [[P:gptoss-pooled:x.c50.c50]] of the time.` → `under Eq. (1) this corresponds to treating a bare assertion as right about [[P:gptoss-pooled:x.c50.c50]] of the time.`
7. `slides.html` l.77: `wrong answers stated at ≥90% resist <i>valid</i> correction.` → `wrong answers stated at ≥90% were corrected less often (80% and 42% vs. 100% below 90%).`
8. `slides.html` l.49: `[[NQ]] MMLU-Pro questions × 4 open-weight models` → `[[NQ]]-question stratified MMLU-Pro sample (43–129 valid items per model) × 4 open-weight models`
9. `report.html` §2: `which would make progressive sycophancy look like recognition.` → `without which progressive sycophancy can look like recognition.`
10. `fliprate/README.md` l.18: `   - defense gap` → `   - confidence–retention gap`; l.19: `AUROC(confidence → keeps answer)` → `AUROC(confidence → does not adopt the asserted answer)` (needs a push)
11. `fliprate/src/make_cover.py` l.30: `fontsize=36` → `fontsize=31` (title clipped). Regenerate the image and copy it to `Submission/project_image.png`.

File: `fliprate/report/FINAL_CHECK.md`. **MUST fixes: 2.**
