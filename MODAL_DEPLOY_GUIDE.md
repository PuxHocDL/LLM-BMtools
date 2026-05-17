# Modal Deployment & Evaluation Guide

## Bước 1: Deploy Models trên Modal

Chạy lần lượt 3 lệnh sau để deploy models lên Modal:

```bash
# 1. Deploy Granite-3.3-8b
modal serve deploy_granite.py

# 2. Deploy Llama-4-maverick-17b  
modal serve deploy_llama4.py

# 3. Deploy GPT-oss-20b
modal serve deploy_gptoss.py
```

Sau khi deploy, Modal sẽ in URL dạng:
```
https://<workspace>--<app-name>--serve/v1
```

## Bước 2: Cập nhật config.yaml

Copy URL từ Modal và cập nhật vào `config_modal.yaml`:

```yaml
agents:
  granite-33-8b:
    model: ibm-granite/granite-3.3-8b-instruct
    api_base: https://<your-granite-url>--serve/v1
    api_key: auto
    temperature: 0.0
    max_steps: 16
    timeout: 60.0

  llama-4-maverick-17b:
    model: meta-llama/Llama-4-Maverick-17B-128E-Instruct
    api_base: https://<your-llama-url>--serve/v1
    api_key: auto
    temperature: 0.0
    max_steps: 16
    timeout: 60.0

  gpt-oss-20b:
    model: openai/gpt-oss-20b
    api_base: https://<your-gptoss-url>--serve/v1
    api_key: auto
    temperature: 0.0
    max_steps: 16
    timeout: 60.0
```

**Lưu ý:** Thay thế `https://<your-xxx-url>--serve/v1` bằng URL thực tế từ Modal.

## Bước 3: Chạy Evaluation

Sau khi đã deploy và cập nhật config:

```bash
# Test nhanh với 10 samples
python main.py --dataset tooljson --model granite-33-8b --limit 10

# Chạy full evaluation (2067 samples)
python main.py --dataset tooljson --model granite-33-8b --workers 10

# Lặp lại cho các model khác
python main.py --dataset tooljson --model llama-4-maverick-17b --workers 10
python main.py --dataset tooljson --model gpt-oss-20b --workers 10
```

## File Đã Tạo

| File | Mô tả |
|---|---|
| `deploy_granite.py` | Deploy Granite-3.3-8b |
| `deploy_llama4.py` | Deploy Llama-4-maverick-17b |
| `deploy_gptoss.py` | Deploy GPT-oss-20b |
| `config_modal.yaml` | Config template (cần cập nhật URL) |

## Cấu hình GPU

- **Granite-3.3-8b**: `a10g` (≈24GB VRAM)
- **Llama-4-maverick-17b**: `a100-80gb` (≈80GB VRAM)
- **GPT-oss-20b**: `a100-40gb` (≈40GB VRAM)

NếuModal không có GPU type này, có thể dùng:
- `a100-40gb`, `a100-80gb`
- `h100`, `h100-80gb`

## Model IDs chính xác (Hugging Face)

| Model | Hugging Face ID |
|---|---|
| Granite-3.3-8b | `ibm-granite/granite-3.3-8b-instruct` |
| Llama-4-maverick-17b | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
| GPT-oss-20b | `openai/gpt-oss-20b` |
