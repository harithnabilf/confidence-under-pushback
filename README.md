# Do Language Models Defend What They Claim to Know?

**Stated confidence under argument-free user pushback.**
This project was built for the [AI x Epistemics Research Sprint](https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15) (Apart Research, Nov 13–15, 2026), Track 1: Model Epistemics & Decision-Making Evals.
Author: Harith Nabil bin Farouk (National University of Malaysia).

All code in this repo was written during the sprint. There is no preexisting code.

## What it does
1. Asks a model an MMLU-Pro question. The model answers and states a confidence from 0 to 100.
2. Branches the conversation:
   - **Neutral re-ask**: "Are you sure?"
   - **C2W**: the model was right, and the user pushes a random wrong option.
   - **W2C**: the model was wrong, and the user pushes the correct option.
   - **W2W**: the model was wrong, and the user pushes a different wrong option (plausibility control).
3. Each pushback branch is a fixed 3-step ladder with no argument content: doubt, then a claimed textbook, then a claimed professor's authority.
4. Scores whether stated confidence predicts which answers the model defends:
   - confidence–retention gap
   - AUROC(confidence → does not adopt the asserted answer)
   - discernment (W2C − C2W, Fisher exact test) and latent discernment (W2C − W2W, paired, exact McNemar test)
   - a logistic threshold c50: the stated confidence at which yielding and holding are equally likely. Under the Bayesian rule "switch iff c < r", c50 estimates the reliability the model implicitly assigns to an unsupported user.

## Reproduce
```bash
pip install requests numpy matplotlib
echo GROQ_API_KEY=... > ../.env      # free-tier Groq key; the runner reads ../.env
python -I src/fetch_data.py           # samples 201 MMLU-Pro test items -> data/questions.jsonl
python -I src/run_experiment.py --n 201 --workers 2    # resumable; writes results/raw/<model>.jsonl
python -I src/analyze.py              # results/tables.md, results/summary.json, figures/*.png
python -I src/render_report.py        # report/Confidence_Under_Pushback_Report.pdf (+ slides)
```
Models used (all via the Groq free tier, temperature 0): `openai/gpt-oss-120b`, `openai/gpt-oss-20b` (reasoning effort low), `qwen/qwen3.8-27b`, `allam-2-7b`.

## Layout
- `src/fetch_data.py`: stratified MMLU-Pro sample from the Hugging Face datasets-server API
- `src/run_experiment.py`: the protocol (prompts at the top of the file)
- `src/analyze.py`: metrics, bootstrap CIs and figures
- `src/extra_stats.py`: exact tests (Fisher, McNemar), item-clustered AUROCs and the logistic threshold c50 (added after internal peer review)
- `src/render_report.py`: fills the report template with computed numbers and prints the PDF (Edge headless)
- `results/raw/*.jsonl`: one record per (model, question), with every call's full text, parsed answer, confidence and token usage
- `report/`: the sprint report (HTML source and PDF)

## License
Code: MIT. MMLU-Pro questions remain under their original license (TIGER-Lab/MMLU-Pro).
