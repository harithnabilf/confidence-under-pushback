# Confidence You Won't Defend

**Do LLMs act on their own stated confidence under escalating user pushback?**
This project was built for the [AI x Epistemics Research Sprint](https://apartresearch.com/sprints/ai-x-epistemics-research-sprint-2026-11-13-to-2026-11-15) (Apart Research, Nov 13–15, 2026), Track 1: Model Epistemics & Decision-Making Evals.
Author: Harith Nabil (National University of Malaysia).

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
   - defense gap
   - AUROC(confidence → keeps answer)
   - discernment (W2C − C2W) and latent discernment (W2C − W2W)
   - Bayesian threshold consistency: "switch iff c < r", which yields an implied user reliability r*

## Reproduce
```bash
pip install requests numpy matplotlib
echo GROQ_API_KEY=... > ../.env      # free-tier Groq key; the runner reads ../.env
python -I src/fetch_data.py           # samples 201 MMLU-Pro test items -> data/questions.jsonl
python -I src/run_experiment.py --n 201 --workers 2    # resumable; writes results/raw/<model>.jsonl
python -I src/analyze.py              # results/tables.md, results/summary.json, figures/*.png
```
Models used (all via the Groq free tier, temperature 0): `openai/gpt-oss-120b`, `openai/gpt-oss-20b` (reasoning effort low), `qwen/qwen3.8-27b`, `allam-2-7b`.

## Layout
- `src/fetch_data.py`: stratified MMLU-Pro sample from the Hugging Face datasets-server API
- `src/run_experiment.py`: the protocol (prompts at the top of the file)
- `src/analyze.py`: metrics, bootstrap CIs and figures
- `results/raw/*.jsonl`: one record per (model, question), with every call's full text, parsed answer, confidence and token usage
- `report/`: the sprint report (HTML source and PDF)

## License
Code: MIT. MMLU-Pro questions remain under their original license (TIGER-Lab/MMLU-Pro).
