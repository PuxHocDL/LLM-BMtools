# Enhancing LLM Processing of JSON Tool Outputs

LLMs increasingly call external REST tools and receive back JSON responses tens
of thousands of characters long, from which they must extract an answer. This
repository evaluates how well models do that on the **ToolJSON** benchmark, and
implements three independent, plug-in improvements over zero-shot prompting.

None of the methods requires fine-tuning; each one targets a different stage of
the pipeline — before, during, and after LLM reasoning.

## Methods

| Method | Stage | Idea |
|---|---|---|
| **(I) JSON Pruning** | Before the LLM | `HeuristicPlus` deterministically scores and prunes the JSON tree so only query-relevant fields remain (≈0.2% of the original size), within a character budget. |
| **(II) Plan-and-Solve** | During reasoning | The model writes one Python script: a `PLAN` (as `#` comments) then `SOLVE` code; the script is executed and the answer comes from `print()`. |
| **(III) Self-Correction** | After code generation | Generate code → execute → classify the error (OK / hard error / empty / format mismatch) → give directed feedback → regenerate, for up to 3 rounds. |

Methods (II) and (III) both run on the pruned JSON from (I).

## Repository layout

```
main.py                  Single entry point
config.example.yaml      Model endpoints — copy to config.yaml
requirements.txt
core/
  llm_client.py          OpenAI-compatible LLM client
  metrics.py             EM / Contains / F1 / LLM-as-a-judge
data_loaders/
  base_loader.py
  tooljson.py            Baseline ToolJSON loader
  methods.py             Loaders for the three methods
evaluators/
  evaluator.py           Concurrent evaluation runner
extensions/              The three improvements
  code_exec.py           Shared sandboxed code execution
  pruning/               (I)  HeuristicPlus JSON pruning
  plan_solve/            (II) Plan-and-Solve code generation
  self_correction/       (III) Self-correction debug loop
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure model endpoints:
   ```bash
   cp config.example.yaml config.yaml   # then fill in api_base / api_key
   ```
   Each agent is an OpenAI-compatible endpoint (vLLM, Modal, OpenAI, ...).
3. Place the ToolJSON benchmark under `data/toolJSONprocessing/` (gitignored;
   the QA pairs are read from `generate_qa_pairs/data/qa_pairs` and the raw API
   responses from `generate_qa_pairs/data/api_responses`).

## Usage

```bash
python main.py --method baseline        --model granite --judge llama
python main.py --method pruning         --model granite
python main.py --method plan_solve      --model granite
python main.py --method self_correction --model gptoss
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--method` | `baseline` | `baseline`, `pruning`, `plan_solve`, `self_correction` |
| `--model` | `granite` | Agent model name from `config.yaml` |
| `--judge` | `llama` | Judge model name from `config.yaml` |
| `--limit` | all | Evaluate only the first N samples (dry-run) |
| `--workers` | 5 | Concurrent worker threads |
| `--budget` | 10000 | HeuristicPlus pruning character budget |
| `--skip-judge` | off | Skip the LLM-as-a-judge metric |

Per-sample results and a summary are written to `results/`.
