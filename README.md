# Enhancing LLM Processing of JSON Tool Outputs

An evaluation pipeline and three plug-in improvements for the task of
**answering questions over large JSON tool outputs** with Large Language Models.

> Want the step-by-step "I've never run an LLM benchmark before" walkthrough?
> Open [`docs/setup.html`](docs/setup.html) in any browser, or read it online
> at https://puxhocdl.github.io/LLM-BMtools/ (auto-deployed on every push).

---

## 1. Background

LLMs are increasingly connected to external tools through REST APIs. When a tool
is called, the model gets back a JSON response that can be **tens of thousands of
characters long** and must extract the right information to answer the user.

Most prior work focuses on *tool selection* and *generating the tool call*, and
overlooks the final step — *processing the tool output*. The base study,
*"How Good Are LLMs at Processing Tool Outputs?"* (Kate et al., 2025),
shows that even strong models stay below 90% accuracy on this task using plain
zero-shot prompting.

This project takes that gap as a starting point and adds three independent,
**fine-tuning-free** improvements, each acting at a different stage of the
pipeline — before, during, and after LLM reasoning.

## 2. Benchmark

The **ToolJSON** benchmark frames the problem as QA over JSON responses: each
sample is a triple *(JSON response, natural-language question, gold answer)*
drawn from 6 real RapidAPI endpoints across travel, finance, and e-commerce.

| Endpoint | Samples |
|---|---:|
| Booking — Search Hotels | 635 |
| Booking — Search Car Rentals | 491 |
| Booking — Get Seat Map | 283 |
| Booking — Get Room List With Availability | 163 |
| SEC — Filings | 267 |
| RealProducts — Shoes | 228 |
| **Total** | **2,067** |

Each JSON response averages 24K–74K characters. Questions fall into three task
types: **Extractive** (read one field), **Filtering** (select records by a
condition), and **Aggregation** (combine many records).

## 3. Methods

| Method | Pipeline stage | Idea |
|---|---|---|
| **(I) JSON Pruning** | Before the LLM | `HeuristicPlus` deterministically scores every JSON leaf against the question, then keeps only the most relevant records within a character budget — reducing the input to roughly 0.2% of its original size while keeping valid JSON. |
| **(II) Plan-and-Solve** | During reasoning | The model writes a single Python script structured as a `PLAN` (`#` comments naming fields, filters and aggregations) followed by `SOLVE` code; the script is executed and the answer is whatever it `print()`s. |
| **(III) Self-Correction** | After code generation | Generate code → execute → classify the error (OK / hard error / empty output / format mismatch) → return directed feedback → regenerate, for up to 3 rounds; stops early if two rounds produce near-identical code. |

Methods (II) and (III) both operate on the pruned JSON produced by (I).

### HeuristicPlus pruning, in brief

1. **Query analysis** — extract keywords, quoted phrases, and the inferred
   answer intent (count / price / rating / id / policy / ...).
2. **Flattening** — turn the JSON tree into a list of leaf nodes, each with its
   JSONPath, raw value, and tokenized representation.
3. **Heuristic scoring** — score every leaf:
   `s = α·match + β·intent + γ·anchor − δ·noise`, where *anchor* rewards
   structural keys (`id`, `name`, `title`) and *noise* penalizes long text blobs.
4. **Record selection & reconstruction** — group leaves by parent record, rank
   records, and rebuild a compact, valid JSON within the budget.

## 4. Repository layout

```
main.py                      Entry point for the three improvement methods
config.example.yaml          Model endpoints — copy to config.yaml
requirements.txt
core/
  llm_client.py              OpenAI-compatible LLM client
  metrics.py                 EM / Contains / F1 / LLM-as-a-judge
data_loaders/
  base_loader.py             Abstract loader interface
  tooljson.py                Baseline ToolJSON loader
  methods.py                 Loaders for the three methods
evaluators/
  evaluator.py               Concurrent evaluation runner
extensions/                  The three improvements
  code_exec.py               Shared sandboxed code execution
  pruning/                   (I)   HeuristicPlus JSON pruning
  plan_solve/                (II)  Plan-and-Solve code generation
  self_correction/           (III) Self-correction debug loop
baseline/                    Self-contained copy of the paper baseline
  run_baseline.py            Run one (setup, model) baseline pair
  run_evaluation.py          Score the baseline predictions
  README.md                  Beginner-friendly walkthrough
  codegen_scripts/           Prompt templates + code executor
  experimental_scripts/      Original qa_inference / qa_evaluation
  generate_qa_pairs/tasks/   Per-endpoint gold-answer logic
docs/
  setup.html                 Full step-by-step setup guide as HTML docs
  index.html                 Redirect landing page
.github/workflows/
  deploy-docs.yml            CI → GitHub Pages
  ci.yml                     Syntax + HTML lint
data/toolJSONprocessing/     Upstream package (Kate et al., 2025)
  generate_qa_pairs/data/    QA pairs, API responses, schemas
```

## 5. Quick start (for a first-time user)

> If you've never touched an LLM benchmark before, follow
> [`docs/setup.html`](docs/setup.html) — it walks you through everything from
> installing Python to reading the results table. The summary below assumes
> you're already comfortable with Python and HTTP APIs.

```bash
git clone <repo-url>
cd LLM-BMtools
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## 6. Configuration

Each model is an OpenAI-compatible endpoint (vLLM, Modal, OpenAI, Azure …).

```bash
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and fill in `api_base` and `api_key` for each agent.
`config.yaml` is git-ignored so credentials are never committed.

```yaml
agents:
  granite:
    model: ibm-granite/granite-3.3-8b-instruct
    api_base: https://YOUR_ENDPOINT/v1
    api_key: YOUR_API_KEY
    temperature: 0.0
    timeout: 180.0
```

### 6.1. Provisioning the LLMs with `m-gpux serve` (recommended)

If you don't already have an OpenAI-compatible endpoint, the easiest way to
get one is the [`m-gpux`](https://github.com/PuxHocDL/m-gpux) CLI, which
deploys a model on [Modal](https://modal.com) as a vLLM-backed,
OpenAI-compatible API with bearer-token auth. Each `m-gpux serve deploy`
creates one Modal app per profile, so the standard workflow is **one profile
per model**:

```bash
pip install m-gpux modal
modal setup                                 # one-time Modal login

# Create one profile per model so the three apps don't collide
m-gpux account add        # name it "granite"
m-gpux account add        # name it "gptoss"
m-gpux account add        # name it "devstral"

# Provision each model (interactive — pick the HF id under "(custom)")
m-gpux account switch granite  && m-gpux serve deploy
m-gpux account switch gptoss   && m-gpux serve deploy
m-gpux account switch devstral && m-gpux serve deploy
```

Each `serve deploy` prints a persistent URL and an API key, e.g.:

```
api_base : https://granite--m-gpux-llm-api-serve.modal.run/v1
api_key  : mgpx_<random>
model    : ibm-granite/granite-3.3-8b-instruct
```

Drop those into `config.yaml`:

```yaml
agents:
  granite:
    model:    ibm-granite/granite-3.3-8b-instruct
    api_base: https://granite--m-gpux-llm-api-serve.modal.run/v1
    api_key:  mgpx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    temperature: 0.0
    timeout: 600.0
  gptoss:
    model:    openai/gpt-oss-20b
    api_base: https://gptoss--m-gpux-llm-api-serve.modal.run/v1
    api_key:  mgpx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    temperature: 0.0
    timeout: 600.0
  devstral:
    model:    mistralai/Devstral-Small-2507
    api_base: https://devstral--m-gpux-llm-api-serve.modal.run/v1
    api_key:  mgpx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    temperature: 0.0
    timeout: 600.0
```

Useful follow-ups:

```bash
m-gpux serve dashboard     # live latency / request metrics
m-gpux serve logs          # tail vLLM logs
m-gpux serve warmup        # pre-warm before a benchmark sweep
m-gpux serve keys create   # rotate API keys
m-gpux stop                # stop the current profile's serve app
m-gpux billing usage --days 7 --all
```

For the `baseline/` scripts (which read `LLM_PROVIDER` from `baseline/.env`),
the same Modal URLs work — just set `LLM_PROVIDER=openai` and point the OpenAI
client at the `api_base` / `api_key` pair above.

## 7. Data

Place the ToolJSON benchmark under `data/` (git-ignored):

```
data/toolJSONprocessing/generate_qa_pairs/data/
  qa_pairs/        # one JSON file of QA pairs per endpoint
  api_responses/   # the raw API JSON responses
  schemas/         # JSON schemas per endpoint
```

## 8. Usage

### A — Re-run the paper baseline (`baseline/`)

```bash
cd baseline
cp example.env .env       # fill in LLM_PROVIDER / AZURE_* or use Ollama
pip install -r requirements.txt

# Reproduce one column of the baseline table:
python run_baseline.py --setup answer       --model ibm-granite/granite-3.3-8b-instruct
python run_baseline.py --setup code_schema  --model openai/gpt-oss-20b
python run_baseline.py --setup code         --model mistralai/Devstral-Small-2507

# Score (EM + Contains, optionally LLM-as-a-judge):
python run_evaluation.py
python run_evaluation.py --judge meta-llama/llama-3-3-70b-instruct
```

See [`baseline/README.md`](baseline/README.md) for the line-by-line walkthrough.

### B — Run the three enhancement methods (root)

```bash
# Baseline — raw JSON, answer directly
python main.py --method baseline        --model granite --judge llama

# (I) JSON pruning
python main.py --method pruning         --model granite

# (II) Pruning + Plan-and-Solve
python main.py --method plan_solve      --model granite

# (III) Pruning + Self-Correction
python main.py --method self_correction --model gptoss

# Quick dry-run on the first 20 samples, no judge
python main.py --method plan_solve --model granite --limit 20 --skip-judge
```

### Options (main.py)

| Flag | Default | Description |
|---|---|---|
| `--method` | `baseline` | `baseline`, `pruning`, `plan_solve`, `self_correction` |
| `--model` | `granite` | Agent model name from `config.yaml` |
| `--judge` | `llama` | Judge model name from `config.yaml` |
| `--limit` | all | Evaluate only the first N samples (dry-run) |
| `--workers` | 5 | Concurrent worker threads |
| `--budget` | 10000 | HeuristicPlus pruning character budget |
| `--skip-judge` | off | Skip the LLM-as-a-judge metric |

Per-sample results (CSV + JSON) and a summary are written to `results/`.

## 9. Metrics

| Metric | Description |
|---|---|
| **Exact Match (EM)** | Relaxed exact comparison (whitespace, case, list order, number rounding). The strictest metric. |
| **Contains** | EM, or the gold answer appears inside the prediction. An upper bound on accuracy. |
| **F1** | Token-level F1 between prediction and gold answer. |
| **LLM-as-a-judge** | A judge model decides semantic equivalence. The most lenient metric. |

## 10. Baseline results (re-run of Kate et al., 2025)

Reproduced on the full 2,067-sample benchmark using the code under
[`baseline/`](baseline/). The three columns are the three baseline settings
the paper compares: direct answer, code-only, code + JSON schema.

| Model | Code (EM / Judge / Contain) | Code + Schema (EM / Judge / Contain) | Answer (EM / Judge / Contain) |
|---|---|---|---|
| Granite-3.3-8B | 0.325 / 0.41 / 0.33 | 0.451 / 0.53 / 0.46 | **0.519** / 0.61 / 0.57 |
| GPT-OSS-20B | 0.615 / 0.76 / 0.62 | **0.723** / 0.85 / 0.74 | 0.260 / 0.44 / 0.59 |
| Devstral-Small-24B | 0.672 / 0.75 / 0.68 | **0.720** / 0.82 / 0.73 | 0.649 / 0.75 / 0.70 |

For each model, **bold** marks the best EM among the three baseline settings —
the one we use as the "Baseline" row when comparing against our enhancements
in §11.

## 11. Enhancement results

ΔEM is the change in EM, in percentage points, versus that model's best
baseline (the bold cell above).

| Model | Method | EM | Contains | Judge | ΔEM |
|---|---|---:|---:|---:|---:|
| Granite-3.3-8B | **Baseline (Answer)** | 0.519 | 0.570 | 0.607 | — |
| | HeuristicPlus pruning | 0.609 | 0.658 | **0.755** | +9.0 |
| | + Plan-and-Solve | **0.687** | **0.720** | 0.732 | **+16.8** |
| | Self-Correction | 0.612 | 0.644 | 0.677 | +9.3 |
| GPT-OSS-20B | **Baseline (Code + Schema)** | 0.723 | 0.735 | 0.760 | — |
| | HeuristicPlus pruning | 0.672 | 0.683 | 0.773 | −5.1 |
| | + Plan-and-Solve | 0.680 | 0.689 | 0.695 | −4.3 |
| | Self-Correction | **0.771** | **0.790** | **0.824** | **+4.8** |
| Devstral-Small-24B | **Baseline (Code + Schema)** | 0.720 | 0.732 | 0.773 | — |
| | HeuristicPlus pruning | 0.717 | 0.731 | **0.826** | −0.3 |
| | + Plan-and-Solve | 0.715 | 0.714 | 0.749 | −0.5 |
| | Self-Correction | **0.742** | **0.754** | 0.757 | **+2.2** |

**Takeaways.** Plan-and-Solve gives the largest gain to the weakest model
(Granite, +16.8 pp EM) by adding an explicit algorithm-design step before
coding — most useful for Filtering. Self-Correction is the most reliable
improvement for code-trained models (GPT-OSS, Devstral), recovering nearly all
execution errors. For models that already generate strong code, the extra
planning step adds more noise than benefit.

## 12. Authors

- Nguyễn Minh Triết
- Nguyễn Minh Bảo
- Nguyễn Xuân Phúc
