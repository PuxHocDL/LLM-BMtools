# HeuristicPlus: Context-Aware JSON Pruning

This repository implements **HeuristicPlus**, a highly optimized, context-aware JSON pruning method designed to reduce the token count of large API responses (like ToolBench or ComplexFunc outputs) before feeding them into LLMs. This helps prevent context overflow, reduces API costs, and increases the accuracy of LLM reasoning by removing irrelevant noise.

## 🌟 How HeuristicPlus Works

Large API responses often contain thousands of lines of JSON, most of which are irrelevant to the user's specific query. **HeuristicPlus** solves this by deterministically scoring and pruning the JSON tree based on the user's question.

The pruning pipeline consists of the following steps:

### 1. Query Analysis & Tokenization
The system first analyzes the user's question to extract:
- **Keywords/Tokens:** Removes stopwords and normalizes words.
- **Exact Phrases:** Identifies quoted strings or specific multi-word entities (e.g., "size 10", "men shoes").
- **Intent Inference:** Detects if the user is asking for a specific data type like a `count`, `price`, `rating`, `id`, or `policy`.

### 2. JSON Flattening
The deeply nested API JSON response is flattened into a one-dimensional array of **Leaf Nodes**. Each leaf stores:
- Its exact JSON path (e.g., `data.products[0].price.amount`).
- Its literal value.
- Tokenized representations of its path and value.

### 3. Heuristic Scoring
Every leaf node is scored using a weighted heuristic function:
- **Direct Matches:** High score if the leaf's path or value matches query tokens/phrases.
- **Intent Matches:** Bonus points if the leaf path matches inferred answer types (e.g., giving bonus to `price` fields if the user asks for "cost").
- **Anchor Retention:** Small bonuses are given to structural keys like `id`, `name`, `type`, or `title` so the LLM retains the context of what object it is looking at.
- **Noise Penalties:** Heavy penalties are applied to long text blobs (like descriptions or footers) and deep nested paths unless the user's query specifically targets them (like "policy" or "terms").

### 4. Record Selection & Reconstruction
Instead of just keeping the top global leaves (which might result in a broken JSON structure), the pruner group leaves by their parent "records" (e.g., array items).
- It ranks entire records based on their aggregated leaf scores.
- It selects the top $N$ most relevant records.
- For each selected record, it filters out low-scoring leaves while keeping anchor keys.
- Finally, it reconstructs the flattened paths back into a valid, highly compact JSON object.

### 5. Deterministic Summary Extraction (The "Plus")
Before handing the pruned JSON to the LLM, the method attempts to aggressively solve common structural questions (e.g., counting items, finding max ratings, filtering by price) using deterministic python rules over the pruned records. If it finds a confident match, it attaches an `__question_summary__` to the JSON. The LLM can then simply verify and output this candidate, resulting in near-perfect accuracy and blazing-fast generation times.

## 🚀 Usage

You can apply the pruning method to any raw JSON and question pair by calling the `heuristic_plus_prune` function:

```python
from data.toolJSONprocessing.extensions.pruning.heuristic_plus import heuristic_plus_prune

pruned_json_object = heuristic_plus_prune(
    json_obj=raw_api_response_dict,
    question="What is the price of the cheapest shoes?",
    max_chars=10000  # Target character budget
)
```

### Running the Evaluation
To run the evaluation pipeline using the HeuristicPlus pruning method against the dataset:

```bash
python run_ablation_3.py --method heuristic_plus --model qwen --budget 10000
```
*(The budget parameter controls the aggressiveness of the pruning).*
