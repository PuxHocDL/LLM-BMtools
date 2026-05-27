# Baseline experiments — Kate et al. (2025) reproduction

This directory is a **self-contained, runnable copy** of the baseline code from
the paper *"How Good Are LLMs at Processing Tool Outputs?"* (Kate et al., 2025).
We extracted the relevant scripts from the original `data/toolJSONprocessing/`
package so that anyone — including a first-year student new to LLM benchmarking
— can re-run the **three baseline settings** that the rest of our project
compares against.

The three settings, with their EM scores from our re-run on the full 2,067-sample
ToolJSON benchmark:

| Model | **Code** | **Code + Schema** | **Answer (direct)** |
|---|---:|---:|---:|
| Granite-3.3-8B | 0.325 | 0.451 | **0.519** |
| GPT-oss-20B | 0.615 | **0.723** | 0.260 |
| Devstral-Small-24B | 0.672 | **0.720** | 0.649 |

`Code` = LLM writes a Python function over the raw JSON. `Code + Schema` = LLM
also sees the JSON schema. `Answer` = LLM answers the question directly without
writing code.

---

## 0. Prerequisites

You only need three things on your machine:

1. **Python 3.10+** — check with `python --version`.
2. **Git** — to clone this repo.
3. **An LLM endpoint** — any OpenAI-compatible HTTP endpoint will work
   (e.g. vLLM, Modal, Ollama running locally, Azure OpenAI, or OpenAI itself).
   We tested with:
   - `ibm-granite/granite-3.3-8b-instruct`
   - `openai/gpt-oss-20b`
   - `mistralai/Devstral-Small-2507`

You do **not** need a GPU on your laptop. The LLM runs on the endpoint; this
script just sends HTTP requests.

---

## 1. Install (one-time)

Open a terminal in the project root and run:

```bash
# Windows PowerShell
cd baseline
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
cd baseline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you get a `genai.extensions.langchain.chat_llm` import error from
`generate_qa_pairs/tasks/utils.py`, you can either:
- Install `ibm-generative-ai`: `pip install ibm-generative-ai`, **or**
- Edit `generate_qa_pairs/tasks/utils.py` and comment out the
  `from genai.extensions...` line — we only use the OpenAI/Ollama paths.

---

## 2. Get the benchmark data

The QA pairs and raw API responses live under
`../data/toolJSONprocessing/generate_qa_pairs/data/`:

```
generate_qa_pairs/data/
  qa_pairs/        # one JSON file of QA pairs per endpoint
  api_responses/   # the raw API JSON
  schemas/         # the JSON schemas
```

If those folders are missing, follow `data/toolJSONprocessing/generate_qa_pairs/README.md`
to regenerate them, or pull them from the upstream repo.

---

## 3. Tell the scripts where your LLM lives

Copy the example env file and fill it in:

```bash
cp example.env .env
```

Open `.env` and set **one** of the following, depending on which provider you
use. Provider is selected by `LLM_PROVIDER`:

```bash
# Option A — Azure OpenAI (paper's default)
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=sk-...
AZURE_ENDPOINT=https://your-resource.openai.azure.com/openai/deployments/{model_id}/chat/completions?api-version={api_version}

# Option B — Local Ollama (recommended for re-running with open weights)
LLM_PROVIDER=local
# (Ollama auto-discovers on http://localhost:11434)
```

For Ollama, also pull the models first:

```bash
ollama pull granite3.3:8b
ollama pull gpt-oss:20b
ollama pull devstral:24b
```

---

## 4. Run a baseline setting

`run_baseline.py` is a small wrapper around the original `qa_inference.py`. It
takes a `--setup` and a `--model` and writes one predictions JSON per endpoint
to `experimental_scripts/results/predictions/`.

```bash
# Reproduce the "Answer" column (direct prompting) for Granite
python run_baseline.py --setup answer       --model ibm-granite/granite-3.3-8b-instruct

# Reproduce the "Code + Schema" column for GPT-OSS-20B
python run_baseline.py --setup code_schema  --model openai/gpt-oss-20b

# Reproduce the "Code" column for Devstral
python run_baseline.py --setup code         --model mistralai/Devstral-Small-2507
```

**Tip — smoke test first.** Add `--limit 5` to run on just 5 samples per
endpoint and verify the pipeline works end-to-end (~30 samples total) before
launching a full run.

```bash
python run_baseline.py --setup answer --model ibm-granite/granite-3.3-8b-instruct --limit 5
```

`--setup` accepts these aliases (or the original setup-type names from the
paper):

| Alias | Paper setup_type | Table column |
|---|---|---|
| `code` | `code_generation` | **Code** |
| `code_schema` | `code_generation_schema` | **Code + Schema** |
| `answer` | `direct_prompting` | **Answer** |
| `answer_schema` | `direct_prompting_schema` | (not in our table) |

---

## 5. Score the predictions

After inference, run the evaluator. By default it computes EM + Contains for
every predictions file. Pass `--judge <model-id>` to also compute
LLM-as-a-judge.

```bash
# EM + Contains only (fast, offline)
python run_evaluation.py

# Add LLM-as-a-judge using e.g. llama-3-70b
python run_evaluation.py --judge meta-llama/llama-3-3-70b-instruct
```

The summary table at the end of the run is what you compare against the EM
column in the table at the top of this README.

---

## 6. Where the results land

```
baseline/experimental_scripts/results/
  predictions/    # one JSON per (endpoint, model, setup) — raw outputs
  evaluation/     # same files with metrics filled in
```

Each `*_eval*.json` file contains, per sample, the `predicted_answer`,
`gold_answer`, `code_exec_status` (for code setups), and the metrics dict with
`exact_match`, `contains`, optional `llm_as_a_judge`, and `hallucination` keys.

---

## 7. Layout (what each file does)

```
baseline/
  run_baseline.py                          ← run one (setup, model) pair
  run_evaluation.py                        ← score everything in predictions/
  example.env                              ← copy to .env and fill in
  requirements.txt
  codegen_scripts/
    direct_prompting_code.py               ← prompt templates for "answer" setups
    general_code_generation.py             ← prompt templates + executor for "code" setups
  experimental_scripts/
    qa_inference.py                        ← original paper inference loop
    qa_evaluation.py                       ← original paper evaluation loop
    counterfactuals.py                     ← path-filtering used by *_cfx2 setups
  generate_qa_pairs/
    tasks/
      data_structures.py                   ← LongResponseQASample dataclass
      utils.py                             ← LLM provider plumbing (Azure / Ollama / OpenAI)
      evals.py                             ← EM / Contains / LLM-as-a-judge
      base.py
      booking_*.py, SEC_filings.py, ...    ← per-endpoint gold-answer logic
```

---

## 8. Going further

To go beyond the baseline, run the enhanced methods from the repo root:

```bash
cd ..
python main.py --method pruning         --model granite
python main.py --method plan_solve      --model granite
python main.py --method self_correction --model gptoss
```

Those use a different (lighter) LLM plumbing layer via `config.yaml` and
share the same ToolJSON benchmark data. See the top-level `README.md` for the
full comparison table.
