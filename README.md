# Enhancing LLM Processing of JSON Tool Outputs

An evaluation pipeline and three plug-in improvements for the task of
**answering questions over large JSON tool outputs** with Large Language Models.

---

## 1. Background

LLMs are increasingly connected to external tools through REST APIs. When a tool
is called, the model gets back a JSON response that can be **tens of thousands of
characters long** and must extract the right information to answer the user.

Most prior work focuses on *tool selection* and *generating the tool call*, and
overlooks the final step — *processing the tool output*. The base study,
*"How Good Are LLMs at Processing Tool Outputs?"* (Kate et al., EACL 2026),
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
main.py                      Single entry point
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
```

## 5. Installation

```bash
git clone <repo-url>
cd LLM-BMtools
pip install -r requirements.txt
```

Requires Python 3.10+.

## 6. Configuration

Each model is an OpenAI-compatible endpoint (vLLM, Modal, OpenAI, ...).

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

## 7. Data

Place the ToolJSON benchmark under `data/` (git-ignored):

```
data/toolJSONprocessing/generate_qa_pairs/data/
  qa_pairs/        # one JSON file of QA pairs per endpoint
  api_responses/   # the raw API JSON responses
```

## 8. Usage

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

### Options

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

## 10. Results

Results reported for three models on the full benchmark (2,067 samples).
ΔEM is the change in EM, in percentage points, versus that model's baseline.

| Model | Method | EM | Contains | Judge | ΔEM |
|---|---|---:|---:|---:|---:|
| Granite-3.3-8B | Baseline (Answer) | 0.519 | — | 0.60 | — |
| | JSON Pruning | 0.609 | 0.658 | 0.755 | +9.0 |
| | + Plan-and-Solve | **0.687** | **0.720** | 0.732 | +16.8 |
| | Self-Correction | 0.612 | 0.644 | 0.677 | +9.3 |
| GPT-OSS-20B | Baseline (Code+Schema) | 0.723 | — | 0.76 | — |
| | JSON Pruning | 0.672 | 0.683 | 0.773 | −5.1 |
| | + Plan-and-Solve | 0.680 | 0.689 | 0.695 | −4.3 |
| | Self-Correction | **0.771** | **0.790** | **0.824** | +4.8 |
| Devstral-Small-24B | Baseline (Code+Schema) | 0.720 | — | 0.77 | — |
| | JSON Pruning | 0.717 | 0.731 | **0.826** | −0.4 |
| | + Plan-and-Solve | 0.715 | 0.714 | 0.749 | −0.5 |
| | Self-Correction | **0.742** | **0.754** | 0.757 | +2.2 |

**Takeaways.** Plan-and-Solve gives the largest gain to the weakest model
(Granite, +16.8 pp EM) by adding an explicit algorithm-design step before
coding — most useful for Filtering. Self-Correction is the most reliable
improvement for code-trained models (GPT-OSS, Devstral), recovering nearly all
execution errors. For models that already generate strong code, the extra
planning step adds more noise than benefit.

## 11. Authors

- Nguyễn Minh Triết
- Nguyễn Minh Bảo
- Nguyễn Xuân Phúc
