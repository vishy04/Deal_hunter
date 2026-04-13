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

## Problem 10: Default notifier constructed but never assigned

**Symptom**

- `AttributeError: 'NoneType' object has no attribute 'send'` when calling `MessagingAgent().push(...)`, `alert(...)`, or `notify(...)` without passing an explicit `notifier`.

**Root cause**

- Line 18 sets `self.notifier = notifier`, which is `None` when no argument is passed.
- Lines 22-25 call `PushoverNotifier(settings.pushover_user, settings.pushover_token)` but never assign the result back to `self.notifier`.
- The freshly created instance gets discarded immediately. `self.notifier` stays `None`.

**Bad pattern**

```python
self.notifier = notifier

if notifier is None:
    PushoverNotifier(
        settings.pushover_user,
        settings.pushover_token,
    )
else:
    self.notifier = notifier
```

**Fix pattern**

```python
self.notifier = notifier or PushoverNotifier(
    settings.pushover_user,
    settings.pushover_token,
)
```

**Note**

- The `else` branch (`self.notifier = notifier`) also duplicated line 18, so `notifier` was assigned twice on the non-`None` path. The one-liner fix handles both branches and removes the dead code.

## Problem 11: Top-level `litellm` import prevents module from loading

**Symptom**

- `SyntaxError: invalid syntax` when importing `MessagingAgent`, or anything that touches `deal_hunter.agents`. The traceback points at `litellm/main.py`, not at your code.

**Root cause**

- `from litellm import completion` sat at the top of `messaging.py`. Python evaluates top-level imports when the module first loads, so a broken or corrupted `litellm` install (in this case `main.py` started with raw markdown) killed the import chain before any class or function was defined.
- Because `agents/__init__.py` re-exports `MessagingAgent`, every consumer of the `agents` package failed too.

**Bad pattern**

```python
from litellm import completion

class MessagingAgent(Agent):
    ...
    def craft_message(self, ...):
        response = completion(...)
```

**Fix pattern**

```python
class MessagingAgent(Agent):
    ...
    def craft_message(self, ...):
        from litellm import completion
        response = completion(...)
```

**Note**

- The lazy import means `MessagingAgent` loads and its other methods (`push`, `alert`, `notify`) work even when `litellm` is broken. The `SyntaxError` only surfaces when `craft_message` actually runs, which is the only method that needs `completion`.
- After fixing the import location, reinstalling `litellm` (`uv pip install --force-reinstall "litellm==1.82.4"`) repaired the corrupted package and made `craft_message` functional again.

## Problem 12: FrontierAgent hardcoded `gpt-40-mini` typo

**Symptom**

- `openai.NotFoundError: 404 ... model gpt-40-mini does not exist` the first time a real planner run reached `FrontierAgent.price()`.
- Mocked tests passed fine because the OpenAI call was stubbed; the typo only surfaced against a live model.

**Root cause**

- `FrontierAgent.__init__` set `self.MODEL = "gpt-40-mini"` (forty, not four-oh).
- The same class also hardcoded the embedding model name and ignored `settings.frontier_model` / `settings.embedding_model`, so there was no single place to fix the string once you noticed it.

**Bad pattern**

```python
class FrontierAgent(Agent):
    MODEL = "gpt-4o-mini"

    def __init__(self, collection) -> None:
        self.client = OpenAI()
        self.MODEL = "gpt-40-mini"
        self.collection = collection
        self.encoder_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
```

**Fix pattern**

```python
class FrontierAgent(Agent):
    def __init__(self, collection) -> None:
        self.client = OpenAI()
        self.MODEL = settings.frontier_model
        self.collection = collection
        self.encoder_model = SentenceTransformer(settings.embedding_model)
```

**Note**

- A second footgun lived in `price()`: `reasoning_effort="low"` was sent unconditionally, which 400s on non-reasoning models like `gpt-4o-mini`. Fix: read `settings.frontier_reasoning_effort`, lower-case it, and only pass `reasoning_effort` when the value is non-empty and not the sentinel `"none"`.

```python
kwargs = {"model": self.MODEL, "messages": ..., "seed": 42}
effort = (settings.frontier_reasoning_effort or "").strip().lower()
if effort and effort != "none":
    kwargs["reasoning_effort"] = effort
response = self.client.chat.completions.create(**kwargs)
```

## Problem 13: EnsembleAgent hardcoded weights

**Symptom**

- Ensemble predictions moved around even when the underlying sub-agents were deterministic, because rerunning a linear-regression fit on a new held-out split was the only way to update the blend.
- `EnsembleAgent.price()` had two magic floats with no explanation of where they came from.

**Root cause**

- The ensemble weights (`0.8115...` frontier, `0.1884...` specialist) were inlined in the `price()` body. Tuning or A/B-ing meant editing code, and the values drifted out of sync with what was written in any config or doc.

**Bad pattern**

```python
class EnsembleAgent(Agent):
    def __init__(self, collection) -> None:
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()

    def price(self, description: str) -> float:
        rewrite = self.preprocessor.preprocess(description)
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        return frontier * 0.8115124189175806 + specialist * 0.1884875810824194
```

**Fix pattern**

```python
class EnsembleAgent(Agent):
    def __init__(self, collection) -> None:
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()
        self.frontier_weight = settings.ensemble_frontier_weight
        self.specialist_weight = settings.ensemble_specialist_weight

    def price(self, description: str) -> float:
        rewrite = self.preprocessor.preprocess(description)
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        return (
            frontier * self.frontier_weight
            + specialist * self.specialist_weight
        )
```

**Note**

- `settings.ensemble_frontier_weight` and `settings.ensemble_specialist_weight` keep the original fitted values as defaults, so day-one behaviour is identical. Later tuning is one env-var change away.

## Problem 14: Gradio `QueueHandler` leak across timer ticks

**Symptom**

- After the UI had been open for a while, each log message started appearing multiple times in the log pane.
- Memory and CPU climbed steadily even on idle ticks where nothing found a new deal.

**Root cause**

- Every timer tick called `run_with_logging`, which built a fresh `queue.Queue` and called `setup_logging` to attach a new `QueueHandler` to the root logger.
- The previous tick's `QueueHandler` was never removed. Each `logging.info(...)` call fanned out to every handler ever attached, including ones pointing at queues Gradio had already stopped reading.

**Bad pattern**

```python
def setup_logging(log_queue: queue.Queue) -> None:
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter(...))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
```

**Fix pattern**

```python
def setup_logging(log_queue: queue.Queue) -> None:
    root = logging.getLogger()
    for old in [h for h in root.handlers if isinstance(h, QueueHandler)]:
        root.removeHandler(old)
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter(...))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
```

**Note**

- The sweep only removes `QueueHandler` instances, so any other handlers (stream, file) survive across ticks. That matters if you run the UI while also tailing logs in a terminal.

