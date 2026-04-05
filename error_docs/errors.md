# Build Errors Review (Interview Notes)

Quick reference of the major issues I hit while building the pricer service.

## Problem 1: Ephemeral pattern was too slow

**Symptom**

- Inference worked, but requests were very slow in real usage.

**Root cause**

- Model loading happened inside `price()`.
- Every call reloaded model/tokenizer.

**Fix**

- Switched to `@app.cls` + `@modal.enter` setup.
- Load model/tokenizer once per container.
- Use Modal Volume cache for HF downloads.

## Problem 2: `attention_mask` warning during generation

**Symptom (Modal logs)**

- "The attention mask and the pad token id were not set..."
- "The attention mask is not set and cannot be inferred..."

**Root cause**

- `tokenizer.encode(...)` gives only `input_ids`.
- `generate()` had no `attention_mask`.
- With `pad_token_id == eos_token_id`, inference becomes ambiguous.

**Bad pattern**

```python
prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=5)
```

**Fix pattern**

```python
prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # input_ids + attention_mask

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
```

**Note**

- `attention_mask` is a 1/0 map: 1 = real token, 0 = padding.

## Problem 3: CUDA OOM on T4 during setup

**Symptom**

- `torch.OutOfMemoryError` in `setup()` at `AutoModelForCausalLM.from_pretrained(...)`.

**Root cause**

- T4 (16GB VRAM) too tight for Llama-3.1-8B load-time peaks.

**Fix**

- Use hybrid placement (GPU + CPU offload) and explicit memory limits.

```python
self.base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    max_memory={0: "14GiB", "cpu": "64GiB"},
    offload_folder=f"{CACHE_DIR}/offload",
)
```

```python
# Env used to reduce allocator fragmentation
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

**Tradeoff**

- OOM fixed on T4, but slower than full-GPU loading.

## Problem 4: Chroma "rebuilds from scratch" after notebook restart

**Symptom**

- Full tqdm ingest runs again after restarting the kernel, even though `notebooks/products_vectorstore/` already has data.
- Or: on-disk DB looks populated, but queries behave like an empty collection.

**Root cause**

- `chromadb.PersistentClient(path="products_vectorstore")` resolves relative to the **Jupyter kernel’s current working directory**, not relative to the notebook file.
- If cwd is repo root one run and `notebooks/` another, Chroma reads/writes **different folders** (e.g. `deal_hunter/products_vectorstore` vs `deal_hunter/notebooks/products_vectorstore`).
- Using `root_dir = Path.cwd().resolve().parents[0]` only equals the repo root when cwd is **`notebooks/`**; if cwd is the repo root, `parents[0]` is the parent of the repo and paths break.

**Fix**

- Point Chroma at a single absolute path under the repo, e.g. `notebooks/products_vectorstore`, built from a `root_dir` that is detected reliably.

```python
from pathlib import Path

_cwd = Path.cwd().resolve()
root_dir = _cwd if (_cwd / "src").is_dir() else _cwd.parent

DB = str(root_dir / "notebooks" / "products_vectorstore")
chroma_client = chromadb.PersistentClient(path=DB)
```

**Sanity check**

```python
import os
print(os.getcwd())
print(root_dir)
print(Path(DB).resolve())
```

- Expect `Path(DB).resolve()` to end with `.../deal_hunter/notebooks/products_vectorstore` and `root_dir` to be `.../deal_hunter` (with a `src/` directory).

**Note**

- With `PersistentClient`, there is no separate “save”; persistence is the folder passed to `path=`. Reuse the same path after restart, then `get_collection(...)` or `get_or_create_collection` with a `count() > 0` short-circuit to skip re-embedding.
