# Build Errors Review

Quick reference of build errors worth remembering. Covers serving patterns, generation correctness, GPU memory, implicit paths, and agent constructor/API bugs.

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

## Problem 5: `ScannerAgent()` crashes with missing `rss` argument

**Symptom**

- `TypeError: ScannerAgent.__init__() missing 1 required positional argument: 'rss'` when calling `ScannerAgent()` with no arguments.

**Root cause**

- The constructor declared `rss: Rss_Service` as a required positional parameter instead of an optional one with a `None` default.
- The plan called for dependency injection where both `rss` and `openai_client` default to `None`, and the constructor creates instances internally if nothing is passed.

**Bad pattern**

```python
def __init__(self, rss: Rss_Service, openai_client=OpenAI) -> None:
    self.rss = rss
    self.openai = openai_client
```

**Fix pattern**

```python
def __init__(
    self,
    rss: Rss_Service | None = None,
    openai_client: OpenAI | None = None,
) -> None:
    self.rss = rss or Rss_Service()
    self.openai = openai_client or OpenAI()
```

**Note**

- The bad version also set `openai_client=OpenAI` (the class, not an instance). That means `self.openai` would be the class object itself, and calling `self.openai.chat.completions.parse(...)` on it would fail because you need an instance.

## Problem 6: Wrong OpenAI API path (`completions.parse` vs `chat.completions.parse`)

**Symptom**

- `AttributeError` at runtime. The `openai.OpenAI()` client has no `completions.parse` method.

**Root cause**

- The call was written as `self.openai.completions.parse(...)`, missing the `chat` segment. The structured-output parse endpoint lives at `client.chat.completions.parse()`, not `client.completions.parse()`.

**Bad pattern**

```python
completion = self.openai.completions.parse(
    model=self.scanner_model,
    ...
)
```

**Fix pattern**

```python
completion = self.openai.chat.completions.parse(
    model=self.scanner_model,
    ...
)
```

## Problem 7: `scan` sends empty prompt to OpenAI when no deals exist

**Symptom**

- If RSS feeds return zero new deals, `scan()` still builds a prompt and calls OpenAI. That wastes tokens on an empty request and might confuse the model into hallucinating deals.

**Root cause**

- No guard clause after `fetch_deals()`. The method assumed there would always be deals to process.

**Bad pattern**

```python
def scan(self, memory: list[str] | None = None) -> DealSelection | None:
    scraped = self.fetch_deals(memory=memory)
    user_prompt = self.make_user_prompt(scraped=scraped)
    completion = self.openai.chat.completions.parse(...)
    ...
```

**Fix pattern**

```python
def scan(self, memory: list[str] | None = None) -> DealSelection | None:
    scraped = self.fetch_deals(memory=memory)
    if not scraped:
        self.log("No new deals found")
        return None
    user_prompt = self.make_user_prompt(scraped=scraped)
    ...
```

## Problem 8: Constructor accepts `None` but never creates default instances

**Symptom**

- `AttributeError: 'NoneType' object has no attribute 'scrape_feeds'` when calling `ScannerAgent().scan()`.

**Root cause**

- The constructor signature was fixed to accept `None` defaults, but the body just assigns them as-is. Calling `ScannerAgent()` leaves `self.rss` and `self.openai` as `None`, so any method that touches them crashes.

**Bad pattern**

```python
def __init__(self, rss: Rss_Service | None = None, openai_client: OpenAI | None = None):
    self.rss = rss
    self.openai = openai_client
```

**Fix pattern**

```python
def __init__(self, rss: Rss_Service | None = None, openai_client: OpenAI | None = None):
    self.rss = rss or Rss_Service()
    self.openai = openai_client or OpenAI()
```

**Note**

- This is the second half of Problem 5. The signature was fixed but the body wasn't. Half a fix is sometimes worse than no fix because the `TypeError` from Problem 5 was at least obvious; this one only shows up when you actually call `scan()`.

## Problem 9: Empty-deals guard uses `is None` instead of `not`

**Symptom**

- `scan()` sends a blank prompt to OpenAI when RSS returns zero deals, wasting tokens and getting back hallucinated results.

**Root cause**

- `fetch_deals` returns a `list`, so when there are no deals it returns `[]`. The guard `if scraped is None` never fires because `[] is None` is `False`. The empty list slides through to `make_user_prompt` and then to the OpenAI call.

**Bad pattern**

```python
scraped = self.fetch_deals(memory=memory)
if scraped is None:
    self.log("No deals found")
    return None
```

**Fix pattern**

```python
scraped = self.fetch_deals(memory=memory)
if not scraped:
    self.log("No deals found")
    return None
```

**Note**

- `not []` is `True` in Python, so `if not scraped` catches both `None` and empty lists. `is None` only catches the literal `None` object.

