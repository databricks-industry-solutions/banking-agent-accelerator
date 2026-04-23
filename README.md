# Deterministic Stateful Agent with Async Human-in-the-Loop

A reference implementation of a **deterministic, stage-driven LangGraph agent** that pauses for an **asynchronous human-in-the-loop (HIL) step** and resumes automatically when the external result arrives. The reference domain is a **private banking assistant** (account statements, opening deposits, AML-style background checks), but the underlying pattern applies to any regulated workflow that combines LLM-powered natural language handling with deterministic business logic and long-running external dependencies.

## Why this exists

Most agent templates let the LLM decide the next step. That's flexible but non-auditable and unpredictable — unacceptable in regulated industries. This accelerator demonstrates a different approach:

1. **Deterministic state machine** — the LLM handles language (intent classification, field extraction, free-form responses); a LangGraph state machine handles *all* control flow. The stage drives routing, the LLM never does.
2. **Async human-in-the-loop via an endpoint-mediated checkpoint write** — the graph pauses at `WAITING_FOR_BACKGROUND_CHECK` and exits. An external system (compliance, fraud, credit bureau, anything slow) POSTs the result to the agent's `/invocations` endpoint with `custom_inputs.background_check_result`; the handler writes it into the LangGraph checkpoint in Lakebase via `graph.aupdate_state()` and returns immediately without an LLM call. The next user message transparently resumes the graph with the result in state.
3. **Workflow-aware UI** — the chat UI surfaces the current stage, intent, and progress via a sidebar badge and filter, driven end-to-end by a `workflow.state.updated` SSE event.

Together these give you an auditable agent that can safely block on minutes-to-hours-long external dependencies without holding a socket open, while keeping the chat UI responsive.

## Architecture

```
┌────────────────┐     ┌────────────────────────┐     ┌────────────────────┐
│  Next.js Chat  │◄───►│  MLflow AgentServer    │◄───►│  LangGraph state   │
│  UI (workflow- │ SSE │  (@invoke / @stream)   │     │  machine           │
│  aware sidebar)│     └───┬───────────▲────────┘     │  (deterministic    │
└────────────────┘         │           │              │   routing)         │
        ▲                  ▼           │              └─────────┬──────────┘
        │          ┌─────────────────────┐                      │
        │ webhook  │  Lakebase Postgres  │◄─────────────────────┘
        │          │  (LangGraph         │  checkpoint read/write
        │          │  AsyncCheckpointer) │  (handler calls
        │          └─────────────────────┘   graph.aupdate_state
        │                          │         on bg_result POST)
        │                          │
        │                          │ POST /invocations with
        │                          │ custom_inputs.background_check_result
┌───────┴──────────────────────────┴──────┐
│  External async system                  │
│  (compliance / fraud / credit bureau /  │
│   any human approval)                   │
└─────────────────────────────────────────┘
```

Three architectural choices make this work:

| Layer | Choice | Why |
|---|---|---|
| Agent control flow | LangGraph with `_route_by_stage` conditional edge | Every transition is named, testable, and auditable. LLM output never changes the graph's next step. |
| Checkpointing | Databricks Lakebase Postgres via `AsyncCheckpointSaver` | Serverless managed Postgres; survives restarts; supports concurrent async writes from external systems. |
| HIL resume | External system POSTs the result to `/invocations`; the handler calls `graph.aupdate_state()` to write into Lakebase and fires an internal webhook to the chat UI | No long-lived sockets, no polling; the next user message picks up the injected result from the checkpoint. |

For a full state-machine diagram and stage-by-stage walkthrough see [`agent_app/agent_server/README.md`](agent_app/agent_server/README.md).

## Reference implementation: private banking

The included demo implements a private-banking assistant with two intents:

- `GENERATE_ACCOUNT_STATEMENT` — collect `customer_id`, `account_id`, period dates; render an email; send.
- `OPEN_DEPOSIT` — collect `customer_id`, `amount`, `currency`, `term_months`, `payout_account`; **submit a mandatory background check (async HIL)**; wait; on approval render email and send; on denial terminate the workflow.

An 8-stage state machine (`START → CLASSIFY_INTENT → GET_TEMPLATE → ASK_FOR_FIELDS ↔ EXTRACT_FIELDS → LOOKUP_CUSTOMER_EMAIL → CUSTOMER_BACKGROUND_CHECK → WAITING_FOR_BACKGROUND_CHECK → CONFIRM → SEND_EMAIL → DONE`) drives the UI's step-progress badge and filter controls.

All tools are stub implementations (`agent_app/agent_server/tools_stubs.py`) — no real banking backend is required to run the demo end-to-end. Each stub supports a `scenario` keyword to exercise error and edge branches (e.g. `ambiguous_email`, `send_failure`, `template_not_found`).

## Extending to another domain

This is the adaptation path when the pattern fits but banking doesn't:

| Change | Where |
|---|---|
| Rename intents | `_INTENT_LABELS` in `agent_app/agent_server/langgraph_agent.py` |
| Change required fields + labels | `_TEMPLATES`, `_FIELD_LABELS` in `agent_app/agent_server/tools_stubs.py` |
| Swap the HIL step | `submit_background_check` + `customer_background_check_submit_node` / `customer_background_check_resume_node` |
| Change stage numbering / progress labels | `_STAGE_STEP`, `_STEP_MESSAGES` |
| Change the external resume contract | `background_check_result` custom_input in `agent.py` + `POST /api/internal/background-check-received` in `server/src/routes/internal.ts` |
| Rebrand the UI | `client/src/components/app-sidebar.tsx`, `greeting.tsx`, `suggested-actions.tsx` |

The framework — stage-driven routing, LLM-for-language/graph-for-flow split, checkpoint-based async resume, workflow-aware SSE events — stays unchanged.

## Repo layout

```
.
├── agent_app/
│   ├── agent_server/           # MLflow ResponsesAgent + LangGraph state machine
│   ├── e2e-chatbot-app-next/   # Next.js chat UI (workflow-aware sidebar, filters, badges)
│   ├── scripts/                # quickstart, discover-tools, start-app
│   ├── databricks.yml          # Databricks Asset Bundle
│   └── app.yaml                # Databricks App runtime config
└── config/
    └── databricks_resources.json  # Workspace-level placeholders (fill in via quickstart)
```

## Quick start

```bash
cd agent_app

# 1. Provision the Lakebase checkpoint store (and register the app + MLflow
#    experiment resources) via the Databricks Asset Bundle.
databricks bundle deploy

# 2. Authenticate, create the MLflow experiment, wire up .env.
uv run quickstart --lakebase banking-agent-memory

# 3. Start the agent server + chat UI locally on http://localhost:8000.
uv run start-app
```

Full setup, evaluation, and deployment instructions: [`agent_app/README.md`](agent_app/README.md).

## Prerequisites

- Databricks workspace on AWS, Azure, or GCP with Unity Catalog enabled.
- **Foundation Model API** enabled with the `databricks-claude-sonnet-4` endpoint reachable (the agent hardcodes this endpoint; swap the `ChatDatabricks(endpoint=...)` line in `agent_server/agent.py` to use a different one).
- **Databricks Lakebase** available in the workspace. The `databricks bundle deploy` step above provisions the Postgres instance used as the LangGraph checkpoint store.
- Local: `uv`, Node 20 via `nvm`, and **Databricks CLI ≥ 0.298** (older versions fail bundle deploy with `error downloading Terraform: openpgp: key expired`; `brew upgrade databricks/tap/databricks` fixes it).

> **First-run note.** The first time you send a message to the agent, `AsyncCheckpointSaver.setup()` runs the LangGraph checkpoint DDL against Lakebase. That can take 30-60s and will appear to hang if you're testing with `curl --max-time 3`. Subsequent requests return in under a second.

## Dependencies and licenses

### Python (`agent_app/`)

| Package | License |
|---|---|
| mlflow-skinny[databricks] | Apache-2.0 |
| databricks-langchain | Apache-2.0 |
| langgraph | MIT |
| langchain-core | MIT |
| fastapi | MIT |
| uvicorn | BSD-3-Clause |

### Node (`agent_app/e2e-chatbot-app-next/`)

| Package | License |
|---|---|
| react | MIT |
| vite | MIT |
| express | MIT |
| drizzle-orm | Apache-2.0 |
| ai (Vercel AI SDK) | Apache-2.0 |
| @radix-ui/* | MIT |
| tailwindcss | MIT |
| biome | MIT |
| playwright | Apache-2.0 |

Exhaustive lists: `agent_app/pyproject.toml`, `agent_app/e2e-chatbot-app-next/package.json`, plus the `package.json` under each workspace.

## External datasets

None. The reference implementation uses synthetic stub data generated at runtime (see `tools_stubs.py`). Adopters are responsible for the data they connect when replacing stubs.

## License

See [LICENSE](LICENSE).

## Security

Report security concerns per [SECURITY.md](SECURITY.md).

## Project support

This project is provided as-is with no warranty. See the Databricks Industry Solutions contribution guidelines for the support model.
