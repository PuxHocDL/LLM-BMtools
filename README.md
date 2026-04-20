# LLM Tool-Calling Evaluation Pipeline

Đánh giá khả năng tool-calling của các LLM trên 3 benchmark, với 5 chiến lược enhancement + baseline + enhanced mode.

## Cấu trúc project

```
├── main.py                  # Entry point chính
├── config.yaml              # Cấu hình model (API endpoint, key, params)
├── rescore.py               # Chấm lại điểm & in bảng so sánh
├── run_all_strategies.bat   # Chạy toàn bộ pipeline cho 1 model
├── requirements.txt
│
├── core/
│   ├── llm_client.py        # OpenAI-compatible API client (timeout, thinking mode)
│   ├── metrics.py           # EM, F1, Contains, Action Match, BLEU, ROUGE, LLM Judge
│   └── enhancers.py         # JSONPruner, SemanticToolRetriever
│
├── data_loaders/
│   ├── base_loader.py       # Abstract base class
│   ├── toolbench.py         # ToolBench loader
│   ├── complexfunc.py       # ComplexFuncBench loader
│   ├── tooljson.py          # ToolJSON loader
│   ├── enhanced_loaders.py  # Enhanced mode (SemanticRetrieval + JSONPruning)
│   └── strategies.py        # 5 strategies × 3 datasets = 15 loader classes
│
├── evaluators/
│   └── evaluator.py         # Main evaluation loop (concurrent, metrics, judge)
│
├── data/                    # Datasets (không push lên git)
│   ├── ToolBench-Static/
│   ├── ComplexFuncBench.jsonl
│   └── toolJSONprocessing/
│
└── results/                 # Kết quả (JSON + CSV per model × strategy × dataset)
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

Cấu hình model trong `config.yaml`:
```yaml
agents:
  qwen:
    model: Qwen/Qwen3.5-35B-A3B
    api_base: https://your-server/v1
    api_key: sk-xxx
    temperature: 0.0
    max_steps: 16
```

## Cách chạy

### Chạy 1 dataset, 1 model

```bash
# Baseline
python main.py --dataset toolbench --model qwen --limit 100 --workers 10

# Enhanced mode
python main.py --dataset complexfunc --model gemma --limit 50 --workers 5 --enhanced

# Với strategy cụ thể
python main.py --dataset tooljson --model glm --limit 100 --workers 10 --strategy s3_cot
```

### Chạy toàn bộ cho 1 model

```bash
# Windows
run_all_strategies.bat qwen 100 10
run_all_strategies.bat gemma 50 5

# Hoặc manual (Linux/Mac)
for s in s1_prompt s2_compress s3_cot s4_twostage s5_fewshot; do
  python main.py --dataset all --model qwen --limit 100 --workers 10 --strategy $s
done
```

### Chấm lại điểm (không cần gọi API)

```bash
python rescore.py
```

### CLI arguments

| Argument      | Default | Mô tả                                                |
|---------------|---------|-------------------------------------------------------|
| `--dataset`   | *required* | `toolbench`, `complexfunc`, `tooljson`, hoặc `all` |
| `--model`     | `qwen`  | Tên model trong `config.yaml`                        |
| `--judge`     | `qwen`  | Model dùng làm LLM judge                            |
| `--limit`     | `None`  | Số sample tối đa (để test nhanh)                     |
| `--workers`   | `15`    | Số thread song song                                  |
| `--enhanced`  | `False` | Bật Enhanced Mode                                    |
| `--strategy`  | `None`  | Chọn strategy: `s1_prompt` ... `s5_fewshot`          |

## 3 Benchmarks

| Dataset | Mô tả | Samples | Đánh giá |
|---------|--------|---------|----------|
| **ToolBench-Static** | Tool selection: cho user query + danh sách API, model phải chọn đúng tool + params | 1,588 | Action Match, EM |
| **ComplexFuncBench** | Multi-step function calling: model chọn đúng function(s) từ danh sách dài | 1,000 | EM (JSON array), Action Match |
| **ToolJSON** | JSON QA: model đọc JSON lớn và trả lời câu hỏi | 495 | EM, Contains, LLM Judge |

## 5 Enhancement Strategies

### S1 — Prompt Rewrite
Viết lại system prompt với format instructions chặt hơn. Bỏ hack "CRITICAL INSTRUCTION" của baseline. Ép model output đúng format `Action: / Action Input:` hoặc JSON array.

### S2 — Tool Compression
Giữ **tất cả** tools nhưng nén mô tả: chỉ giữ tên + 80 ký tự description đầu + tên params. Giảm token count ~60% mà không mất tool nào.

### S3 — Chain-of-Thought
Thêm hướng dẫn suy luận từng bước trước khi chọn tool. Ví dụ: "Step 1: Identify what the user needs. Step 2: Match to available tools. Step 3: Output action."

### S4 — Two-Stage LLM
Dùng LLM làm preprocessor (stage 1) để lọc tools/data trước, rồi LLM chính (stage 2) chỉ làm việc với context đã thu gọn. Tốn 2× API calls nhưng giảm noise.

### S5 — Few-Shot Examples
Thêm 1 example đúng format vào prompt. Model thấy ví dụ input→output rồi bắt chước.

### Enhanced Mode (existing)
- **ToolBench**: `SemanticToolRetriever` lọc top-3 tools liên quan nhất bằng TF-IDF scoring
- **ToolJSON**: `JSONPruner` rút gọn JSON (lookup hoặc aggregation) để tránh context overflow

## Metrics

| Metric | Mô tả |
|--------|--------|
| **Exact Match (EM)** | So sánh chính xác sau khi trích xuất (Action, JSON array, hoặc final answer sau `</think>`) |
| **Contains** | Ground truth có nằm trong prediction không |
| **F1** | Token-level F1 score |
| **Action Match** | So sánh tên tool/function (bỏ qua arguments) |
| **LLM Judge** | Model judge đánh giá 0/1 dựa trên ngữ nghĩa |
| **BLEU / ROUGE** | N-gram overlap metrics |

## Output

Kết quả lưu trong `results/` với naming convention:
```
{LoaderClassName}_{model}_detailed.json   # Summary + per-sample details
{LoaderClassName}_{model}_detailed.csv    # Bảng CSV
{LoaderClassName}_trace.json              # Error trace log
```

## Phân công cho teammates

Mỗi người chạy model của mình:

```bash
# Người 1: Qwen
run_all_strategies.bat qwen 100 10

# Người 2: Gemma  
run_all_strategies.bat gemma 100 10

# Người 3: GLM
run_all_strategies.bat glm 100 10

# Người 4: LFM
run_all_strategies.bat lfm 100 10
```

Trước khi chạy, cập nhật `config.yaml` với API endpoint đúng cho model của bạn.

Kết quả sẽ tự động lưu riêng theo tên model (VD: `S1_ToolBench_gemma_detailed.json`).

Sau khi tất cả chạy xong, dùng `python rescore.py` để in bảng tổng hợp.
