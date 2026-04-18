# `AutonomousPlanningAgent` — Visual Overview

Companion diagrams for [.cursor/plans/phase_e_beginner_walkthrough_2422ea9e.plan.md](../../.cursor/plans/phase_e_beginner_walkthrough_2422ea9e.plan.md).

The autonomous planner does the same job as the deterministic `PlanningAgent` (scan -> price -> alert) but delegates the ordering to an LLM via OpenAI **tool calling**. This doc shows how the pieces fit.

---

## 1. Landscape — where the autonomous planner sits

```mermaid
flowchart LR
    subgraph UI [Gradio UI - phase G]
        App[App]
    end

    subgraph Framework [DealAgentFramework]
        FW[framework.py]
        Mem[(memory.json)]
        Chroma[(ChromaDB<br/>products_vectorstore)]
    end

    subgraph Planners
        Det[PlanningAgent<br/>deterministic]
        Auto[AutonomousPlanningAgent<br/>phase E]
    end

    subgraph SubAgents [Sub agents - shared]
        Scan[ScannerAgent]
        Ens[EnsembleAgent]
        Msg[MessagingAgent]
    end

    subgraph External
        OAI[(OpenAI API<br/>settings.planner_model)]
        RSS[(RSS feeds)]
        Modal[(Modal Pricer)]
        Push[(Pushover)]
    end

    App -->|run| FW
    FW -->|reads/writes| Mem
    FW -->|get_or_create_collection| Chroma
    FW -->|lazy init| Det
    FW -.-|swap in phase H| Auto

    Det --> Scan
    Det --> Ens
    Det --> Msg
    Auto --> Scan
    Auto --> Ens
    Auto --> Msg
    Auto -->|tool loop| OAI

    Scan --> RSS
    Ens --> Modal
    Msg --> Push
```

---

## 2. Sequence — one `plan()` turn with the LLM driving

```mermaid
sequenceDiagram
    autonumber
    participant FW as DealAgentFramework
    participant AP as AutonomousPlanningAgent
    participant OAI as OpenAI (planner_model)
    participant SC as ScannerAgent
    participant EN as EnsembleAgent
    participant MS as MessagingAgent

    FW->>AP: plan(memory=list[Opportunity])
    AP->>AP: self.memory = [o.deal.url for o in memory]<br/>self.opportunity = None

    loop while not done
        AP->>OAI: chat.completions.create(model, messages, tools)
        OAI-->>AP: choice (finish_reason + message)

        alt finish_reason == "tool_calls"
            AP->>AP: append assistant message
            par one per tool_call
                OAI--)AP: scan_the_internet_for_bargains()
                AP->>SC: scan(memory=self.memory)
                SC-->>AP: DealSelection | None
                AP-->>OAI: model_dump_json() or "No deals found"
            and
                OAI--)AP: estimate_true_value(description)
                AP->>EN: price(description)
                EN-->>AP: float
                AP-->>OAI: "The estimated true value of ... is ..."
            and
                OAI--)AP: notify_user_of_deal(desc, price, value, url)
                AP->>AP: guard: self.opportunity already set?
                AP->>MS: notify(desc, price, value, url)
                MS-->>AP: bool
                AP->>AP: self.opportunity = Opportunity(...)
                AP-->>OAI: "Notification sent ok"
            end
        else finish_reason == "stop"
            AP->>AP: done = True
        end
    end

    AP-->>FW: self.opportunity | None
    FW->>FW: memory.append(result) + write_memory()
```

---

## 3. The `plan()` control loop (flow view)

```mermaid
flowchart TD
    Start([plan memory]) --> Init[self.memory = urls<br/>self.opportunity = None]
    Init --> Seed[messages = system + user]
    Seed --> Call[[openai.chat.completions.create<br/>model, messages, tools]]
    Call --> Finish{finish_reason}

    Finish -- tool_calls --> AppA[messages.append<br/>assistant message]
    AppA --> Dispatch[handle_tool_call<br/>for each tool_call]
    Dispatch --> AppT[messages.extend<br/>tool results]
    AppT --> Call

    Finish -- stop --> LogDone[log completed + reply]
    LogDone --> Ret([return self.opportunity])
```

---

## 4. Tool dispatch — how strings become Python calls

```mermaid
flowchart LR
    Msg["assistant message<br/>tool_calls = [...]"] --> Loop{for each<br/>tool_call}

    Loop --> Name[tool_call.function.name]
    Loop --> Args["args = json.loads<br/>tool_call.function.arguments"]

    Name --> Map{mapping lookup}
    Map -- scan_the_internet_for_bargains --> F1[self.scan_the_internet_for_bargains]
    Map -- estimate_true_value --> F2[self.estimate_true_value]
    Map -- notify_user_of_deal --> F3[self.notify_user_of_deal]

    F1 --> Result
    F2 --> Result
    F3 --> Result
    Args --> Result[tool returns str]

    Result --> Wrap["append<br/>{role: tool, content: str(result), tool_call_id: id}"]
    Wrap --> Loop
```

---

## 5. Per-tool flow

### 5a. `scan_the_internet_for_bargains`

```mermaid
flowchart TD
    A([LLM calls scan]) --> B[log calling scanner]
    B --> C[[ScannerAgent.scan memory=self.memory]]
    C --> D{results?}
    D -- yes --> E[results.model_dump_json]
    D -- no --> F["No deals found"]
    E --> G([return JSON str])
    F --> G
```

### 5b. `estimate_true_value(description)`

```mermaid
flowchart TD
    A([LLM calls estimate]) --> B[log estimating via Ensemble]
    B --> C[[EnsembleAgent.price description]]
    C --> D[float estimate]
    D --> E["return 'The estimated true value of ... is ...'"]
```

### 5c. `notify_user_of_deal(...)` — with single-notify guard

```mermaid
flowchart TD
    Start([LLM calls notify]) --> Guard{self.opportunity set?}
    Guard -- yes --> Skip[log 2nd call ignored]
    Skip --> RetDup(["return 'already sent ignored'"])

    Guard -- no --> LogN[log notifying user]
    LogN --> Msg[[MessagingAgent.notify<br/>desc, price, value, url]]
    Msg --> Build[Deal + discount]
    Build --> Save[self.opportunity = Opportunity deal, estimate, discount]
    Save --> RetOK(["return 'Notification sent ok'"])
```

---

## 6. Tool schemas — the contract the LLM sees

```mermaid
classDiagram
    class scan_function {
        name: scan_the_internet_for_bargains
        description: returns top bargains
        parameters: object with no properties
    }
    class estimate_function {
        name: estimate_true_value
        description: estimate true worth
        parameters.description: string
        required: [description]
    }
    class notify_function {
        name: notify_user_of_deal
        description: send push; only call once
        parameters.description: string
        parameters.deal_price: number
        parameters.estimated_true_value: number
        parameters.url: string
        required: all four
    }

    class ToolWrapper {
        type: function
        function: {...}
    }

    ToolWrapper <|.. scan_function
    ToolWrapper <|.. estimate_function
    ToolWrapper <|.. notify_function
```

---

## 7. Class view — collaborators and inheritance

```mermaid
classDiagram
    class Agent {
        <<base>>
        +GREEN
        +log msg
    }

    class AutonomousPlanningAgent {
        +name: "Autonomous Planning Agent"
        +color = Agent.GREEN
        -scanner: ScannerAgent
        -ensemble: EnsembleAgent
        -messenger: MessagingAgent
        -openai: OpenAI
        -model: str
        -memory: list~str~
        -opportunity: Opportunity | None
        +SYSTEM_MESSAGE
        +USER_MESSAGE
        +scan_function
        +estimate_function
        +notify_function
        +__init__(collection, scanner?, ensemble?, messenger?, openai_client?)
        +scan_the_internet_for_bargains() str
        +estimate_true_value(description) str
        +notify_user_of_deal(desc, price, value, url) str
        +get_tools() list~dict~
        +handle_tool_call(message) list~dict~
        +plan(memory) Opportunity | None
    }

    class PlanningAgent {
        +plan(memory) Opportunity | None
        +run(deal) Opportunity
    }

    class ScannerAgent
    class EnsembleAgent
    class MessagingAgent
    class OpenAI {
        +chat.completions.create
    }
    class Opportunity {
        +deal: Deal
        +estimate: float
        +discount: float
    }

    Agent <|-- AutonomousPlanningAgent
    Agent <|-- PlanningAgent
    AutonomousPlanningAgent --> ScannerAgent : DI
    AutonomousPlanningAgent --> EnsembleAgent : DI
    AutonomousPlanningAgent --> MessagingAgent : DI
    AutonomousPlanningAgent --> OpenAI : DI
    AutonomousPlanningAgent ..> Opportunity : produces
    PlanningAgent ..> Opportunity : produces
    PlanningAgent --> ScannerAgent
    PlanningAgent --> EnsembleAgent
    PlanningAgent --> MessagingAgent
```

Both planners share the same public surface — `plan(memory) -> Opportunity | None` — so `DealAgentFramework` can swap them in phase H without changes to `run()`.

---

## 8. Deterministic vs autonomous — side by side

```mermaid
flowchart LR
    subgraph Deterministic [PlanningAgent]
        D1[scanner.scan] --> D2[run deal for top 5] --> D3[sort by discount] --> D4{best.discount ><br/>deal_threshold?}
        D4 -- yes --> D5[messenger.alert] --> D6([Opportunity])
        D4 -- no --> D7([None])
    end

    subgraph Autonomous [AutonomousPlanningAgent]
        A0[LLM reads system+user prompt] --> A1{LLM decides}
        A1 -- call --> A2[scan/estimate/notify tools]
        A2 --> A1
        A1 -- reply text --> A3([self.opportunity])
    end
```

Same inputs, same sub-agents, same output type. The difference: *who* decides the order — your code, or the model.

---

## 9. Types in, types out

```mermaid
flowchart LR
    FW["DealAgentFramework.memory: list~Opportunity~"] -->|plan memory| AP[AutonomousPlanningAgent]
    AP -->|extract| URLS["list~str~ URLs<br/>for ScannerAgent"]
    URLS --> SC[ScannerAgent.scan]
    SC -->|DealSelection -> JSON| LLM[(OpenAI)]
    LLM -->|JSON args| NOTIFY[notify_user_of_deal]
    NOTIFY --> NEW["Opportunity<br/>Deal + estimate + discount"]
    NEW --> AP
    AP -->|return| FW
```

Key conversion: `list[Opportunity]` comes in, URLs are extracted internally for the scanner, and exactly one `Opportunity` (or `None`) comes out.

---

## 10. Acceptance checklist mapped to diagrams

| Check | Covered by |
| ----- | ---------- |
| `plan` returns `Opportunity` or `None` | Section 2, 3 |
| Tool dispatch matches schemas | Section 4, 6 |
| Single-notify guard fires on 2nd call | Section 5c |
| DI for scanner / ensemble / messenger / openai | Section 7 |
| Same public surface as `PlanningAgent` | Section 7, 8 |
| Framework can swap planners | Section 1 |
