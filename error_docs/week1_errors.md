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

## Rapid-Fire Interview Prep

- **Why `@app.cls` over `@app.function`?** Expensive model load runs once in `@modal.enter`, then reused across calls.
- **Why was ephemeral pattern slow?** Model/tokenizer loaded inside request path.
- **What is `attention_mask`?** 1/0 map: 1 real token, 0 padding.
- **Why did attention warnings appear?** Used `tokenizer.encode(...)` (no mask), then `generate()` without `attention_mask`.
- **Why did T4 OOM happen?** 16GB VRAM couldn’t fit load-time memory peaks for 8B model.
- **How did you fix T4 OOM?** `device_map="auto"` + `max_memory` + `offload_folder` + fp16 + low-cpu-mem loading.
- **Tradeoff of offload fix?** Works on smaller GPU, but inference is slower.
- **One gotcha to remember?** `PYTORCH_CUDA_ALLOC_CONF` must be `expandable_segments:True` (no extra spaces).
