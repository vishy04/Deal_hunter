# `deal_hunter.services` — Visual Overview

Five services live under `src/deal_hunter/services/`:

| File                 | Class / API                         | Role                                              |
| -------------------- | ----------------------------------- | ------------------------------------------------- |
| `rss.py`             | `Rss_Service`                       | Pull RSS feeds, scrape product pages into `ScrapedDeal` |
| `notifications.py`   | `PushoverNotifier`                  | Push mobile notifications via Pushover HTTP API   |
| `preprocessing.py`   | `Preprocessor`                      | Rewrite raw product text into a concise LLM-friendly form |
| `pricer.py`          | `Pricer` (Modal `@app.cls`)         | Remote GPU inference with a LoRA-tuned Llama-3.1-8B |
| `testing.py`         | `Tester` / `evaluate()`             | Offline evaluation of any price predictor, with Plotly chart |

---

## 1. Landscape — how services relate to the rest of the system

```mermaid
flowchart LR
    subgraph Agents
        Scanner[ScannerAgent]
        Specialist[SpecialistAgent]
        Messaging[MessagingAgent]
        Ensemble[EnsembleAgent]
    end

    subgraph Services [src/deal_hunter/services]
        RSS[Rss_Service<br/>rss.py]
        Pre[Preprocessor<br/>preprocessing.py]
        Pricer[Pricer @ Modal<br/>pricer.py]
        Push[PushoverNotifier<br/>notifications.py]
        Test[Tester / evaluate<br/>testing.py]
    end

    subgraph External
        Feeds[(DealNews RSS)]
        Sites[(Product pages<br/>HTML)]
        Groq[(Groq / LiteLLM)]
        Modal[(Modal Cloud GPU<br/>T4)]
        PushoverAPI[(Pushover API)]
    end

    Scanner --> RSS --> Feeds
    RSS --> Sites
    Specialist --> Pricer --> Modal
    Messaging --> Push --> PushoverAPI
    Ensemble -. evaluated by .-> Test
    Pre --> Groq
```

---

## 2. `Rss_Service` — feed fan-in and page scrape

```mermaid
flowchart TD
    Start([scrape_feeds feed_urls, known_urls]) --> LoopFeeds{for each feed_url}
    LoopFeeds --> Parse[feedparser.parse]
    Parse -->|ok| Take[take first max_per_feed entries]
    Parse -->|error| LogF[log parse failure]
    Take --> Collect[all_entries]
    LogF --> Collect

    Collect --> LoopEntries{for each entry}
    LoopEntries --> GetURL[extract URL from links or link]
    GetURL --> Known{url in known_urls?}
    Known -- yes --> Skip[skip]
    Known -- no --> Scrape[scrape_entry]

    Scrape --> Extract[extract summary div.snippet_summary]
    Scrape --> HTTP[requests.get url, timeout=10]
    HTTP -->|ok| Soup[BeautifulSoup parse]
    Soup --> Content{find div.content-section}
    Content -- has Features --> Split[split details / features]
    Content -- no Features --> Details[details = text]
    Content -- missing --> Fallback[use summary as details]
    HTTP -->|fail| LogH[log fetch error] --> Fallback

    Split --> Build[build ScrapedDeal]
    Details --> Build
    Fallback --> Build
    Build --> Trunc[deal.truncate] --> Add[known_urls.add url] --> Sleep[time.sleep delay]
    Skip --> LoopEntries
    Sleep --> LoopEntries
    LoopEntries --> Return([return list of ScrapedDeal])
```

---

## 3. `PushoverNotifier` — notification send path

```mermaid
flowchart TD
    Start([send message, sound]) --> Len{len message > 1024?}
    Len -- yes --> Warn[log over-limit warning]
    Len -- no --> Build
    Warn --> Build[build payload<br/>user, token, message, sound]

    Build --> POST[[POST api.pushover.net/1/messages.json]]
    POST -->|exception| LogErr[log HTTP error] --> RetFalse1([return False])
    POST --> Raise{raise_for_status}
    Raise -- 4xx/5xx --> LogErr
    Raise -- ok --> JSON{response.json}
    JSON -- fails --> LogJSON[log JSON error] --> RetFalse2([return False])
    JSON -- ok --> Status{data.status == 1?}
    Status -- no --> LogRej[log Pushover errors] --> RetFalse3([return False])
    Status -- yes --> LogOK[log sent] --> RetTrue([return True])
```

---

## 4. `Preprocessor` — LLM rewrite via LiteLLM

```mermaid
sequenceDiagram
    participant Caller
    participant P as Preprocessor
    participant LL as litellm.completion
    participant M as groq/openai/gpt-oss-20b

    Caller->>P: preprocess(text)
    P->>P: messages_for(text)<br/>system=SYSTEM_PROMPT<br/>user=text
    P->>LL: completion(model, messages, reasoning_effort)
    LL->>M: HTTPS request
    M-->>LL: chat completion
    LL-->>P: response
    P-->>Caller: choices[0].message.content
```

Fields tracked per instance: `total_input_tokens`, `total_output_tokens`, `total_cost` (scaffolded, not yet populated by `preprocess`).

---

## 5. `Pricer` — Modal-hosted LoRA inference

```mermaid
flowchart TD
    subgraph Local
        Caller[Caller<br/>e.g. SpecialistAgent]
    end

    subgraph Modal [Modal App: pricer]
        direction TB
        Enter["@modal.enter setup"]
        Method["@modal.method price description"]
        Enter --> Tok[AutoTokenizer from_pretrained<br/>Llama-3.1-8B-Instruct]
        Tok --> PadCfg[pad_token = eos<br/>padding_side = right]
        Enter --> Base[AutoModelForCausalLM<br/>4-bit nf4 quant<br/>device_map auto]
        Base --> LoRA[PeftModel.from_pretrained<br/>Vishy08/product-pricer-...]
        LoRA --> Ready((warm container))

        Method --> Prompt["prompt = QUESTION + description + PREFIX"]
        Prompt --> Seed[set_seed 42]
        Seed --> Tokenize[tokenizer prompt -> input_ids, attention_mask]
        Tokenize --> GPU[move tensors to cuda]
        GPU --> Gen[generate max_new_tokens=5]
        Gen --> Decode[tokenizer.decode]
        Decode --> Split[split on 'Price is $']
        Split --> Regex[regex extract float]
        Regex --> Ret([return float or 0.0])
    end

    Caller -- remote call --> Method
    Ready -. served by .- Method

    subgraph Infra
        Vol[(Volume<br/>hf-hub-cache -> /cache)]
        Secret[[Secret<br/>huggingface-secret]]
        GPUhw[[GPU T4<br/>timeout 1800s]]
    end
    Vol -. mounted .-> Enter
    Secret -. env .-> Enter
    GPUhw -. runs .-> Method
```

Key production detail: `min_containers=0` means cold starts are possible. The `@modal.enter` hook pays the model-load cost once per container, then `price()` is fast.

---

## 6. `Tester` / `evaluate` — offline eval loop

```mermaid
flowchart TD
    Start([evaluate predictor, data, size=250]) --> Init[Tester.__init__<br/>allocate numpy arrays<br/>guesses, truths, errors, lche, sles]
    Init --> Run[run]
    Run --> Loop{for i in range size}
    Loop --> One[run_datapoint i]

    One --> Pick[data.iloc i]
    Pick --> Predict[guess = float predictor datapoint]
    Predict --> Metrics[error, log_error, sle, log_cosh]
    Metrics --> Color[color_for error, truth]

    Color -->|green| G[green_count++]
    Color -->|orange| O[orange_count++]
    Color -->|red| R[red_count++]
    G --> Store
    O --> Store
    R --> Store[store into arrays + colors]
    Store --> Loop

    Loop -- done --> Report[report]
    Report --> Print[print stats with ANSI colors]
    Report --> Chart[chart via Plotly<br/>±20 / ±40 bands + y=x]
    Chart --> End([done])

    subgraph ColorRule [color_for]
        C1{truth <= 0}
        C1 -- yes --> O1[orange]
        C1 -- no --> C2{error < 40 or error/truth < 0.2}
        C2 -- yes --> G1[green]
        C2 -- no --> C3{error < 80 or error/truth < 0.4}
        C3 -- yes --> O2[orange]
        C3 -- no --> R1[red]
    end
```

Guardrail in `__init__`: if the caller passes `(DataFrame, callable)` by mistake, the args are swapped with a `UserWarning`.

---

## 7. Class view — inheritance and boundaries

```mermaid
classDiagram
    class Agent {
        <<base>>
        +name
        +color
        +log msg
    }

    class Rss_Service {
        +extract html_snippet str
        +scrape_entry entry dict ScrapedDeal
        +scrape_feeds feed_urls, known_urls, max_per_feed, delay list~ScrapedDeal~
    }

    class PushoverNotifier {
        -user_key
        -token
        -url
        +send message, sound bool
    }

    class Preprocessor {
        -model_name
        -reasoning_effort
        -total_input_tokens
        -total_output_tokens
        -total_cost
        +messages_for text list~dict~
        +preprocess text str
    }

    class Pricer {
        <<Modal @app.cls>>
        -tokenizer
        -base_model
        -fine_tuned_model
        +setup
        +price description float
    }

    class Tester {
        -predictor
        -data
        -size
        +run
        +report
        +chart title
        +test function, data classmethod
    }

    Agent <|-- Rss_Service
    Agent <|-- PushoverNotifier
    Rss_Service ..> ScrapedDeal : produces
```

`Preprocessor`, `Pricer`, and `Tester` deliberately do **not** inherit from `Agent` — they are infrastructure, not agents in the messaging sense. `Rss_Service` and `PushoverNotifier` do inherit from `Agent` only to reuse the colored `log()` helper.
