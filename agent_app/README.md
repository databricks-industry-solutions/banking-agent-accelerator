# Responses API Agent

This template defines a conversational agent app. The app comes with a built-in chat UI, but also exposes an API endpoint for invoking the agent so that you can serve your UI elsewhere (e.g. on your website or in a mobile app).

The agent in this template implements the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) interface. It has access to a single tool; the [built-in code interpreter tool](https://docs.databricks.com/aws/en/generative-ai/agent-framework/code-interpreter-tools#built-in-python-executor-tool) (`system.ai.python_exec`) on Databricks. You can customize agent code and test it via the API or UI.

The agent input and output format are defined by MLflow's ResponsesAgent interface, which closely follows the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) interface. See [the MLflow docs](https://mlflow.org/docs/latest/genai/flavors/responses-agent-intro/) for input and output formats for streaming and non-streaming requests, tracing requirements, and other agent authoring details.

## Build with AI Assistance

We recommend using AI coding assistants (Claude Code, Cursor, GitHub Copilot) to customize and deploy this template. Agent Skills in `.claude/skills/` provide step-by-step guidance for common tasks like setup, adding tools, and deployment. These skills are automatically detected by Claude, Cursor, and GitHub Copilot.

## Quick start

```bash
# 1. Provision the Lakebase checkpoint store + register the app and MLflow
#    experiment resources via the Databricks Asset Bundle. First run takes
#    5-10 min while the Lakebase instance comes up.
databricks bundle deploy

# 2. Configure local authentication, create the MLflow experiment, wire up
#    .env, and validate that the Lakebase instance from step 1 is reachable.
uv run quickstart --lakebase banking-agent-memory

# 3. Start the agent server + chat UI locally.
uv run start-app
```

`uv run quickstart` does the following:

1. Verifies `uv`, `nvm`/Node, and `databricks` CLI are installed.
2. Walks you through Databricks OAuth (or uses an existing profile if you pass `--profile`).
3. Creates an MLflow experiment under `/Users/<you>/agents-on-apps` and writes the ID to `.env`.
4. Validates that the Lakebase instance exists (the quickstart does *not* create it — that's what `databricks bundle deploy` is for).
5. Wires chat-history Postgres env vars (`PGHOST`, `PGUSER`, `PGDATABASE`, `PGPORT`) so the chatbot UI persists chats across restarts. Pass `--no-chat-history` to skip and run in ephemeral mode.

`uv run start-app` then installs chatbot dependencies, runs Drizzle migrations to create the `ai_chatbot` schema, and starts the agent server + chat UI at http://localhost:8000.

> **Requires Databricks CLI ≥ 0.298.** Older versions fail `databricks bundle deploy` with `error downloading Terraform: openpgp: key expired`. Upgrade with `brew upgrade databricks/tap/databricks` (macOS) or the install script from the [Databricks CLI docs](https://docs.databricks.com/dev-tools/cli/install.html) (Linux).

> **First-invocation delay.** The very first agent request triggers `AsyncCheckpointSaver.setup()`, which creates the LangGraph checkpoint tables in Lakebase. Expect 30-60s for that first request; subsequent requests return in under a second. If you're smoke-testing with `curl`, pass `--max-time 90` or longer on the first call.

> **Important:** `uv run quickstart` requires the Lakebase instance to already exist — it validates rather than creates. Run `databricks bundle deploy` first so the `banking-agent-memory` instance is provisioned. If you're customising the instance name, pass it to `--lakebase <name>` and update the `database_instances.agent_memory.name` field in `databricks.yml` to match.

**Next steps**: see [modifying your agent](#modifying-your-agent) to customize and iterate on the agent code.

## Architecture

For a detailed walkthrough of how the agent server works internally (state machine, tool stubs, request flow, and testing), see [`agent_server/README.md`](agent_server/README.md). This includes the async background check flow, which uses checkpoint injection and internal webhooks to pause the graph while waiting for an external result and resume automatically when it arrives.

## Manual local development loop setup

1. **Set up your local environment**
   Install `uv` (python package manager), `nvm` (node version manager), and the Databricks CLI:

   - [`uv` installation docs](https://docs.astral.sh/uv/getting-started/installation/)
   - [`nvm` installation](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating)
     - Run the following to use Node 20 LTS:
       ```bash
       nvm use 20
       ```
   - [`databricks CLI` installation](https://docs.databricks.com/aws/en/dev-tools/cli/install)

2. **Set up local authentication to Databricks**

   In order to access Databricks resources from your local machine while developing your agent, you need to authenticate with Databricks. Choose one of the following options:

   **Option 1: OAuth via Databricks CLI (Recommended)**

   Authenticate with Databricks using the CLI. See the [CLI OAuth documentation](https://docs.databricks.com/aws/en/dev-tools/cli/authentication#oauth-user-to-machine-u2m-authentication).

   ```bash
   databricks auth login
   ```

   Set the `DATABRICKS_CONFIG_PROFILE` environment variable in your .env file to the profile you used to authenticate:

   ```bash
   DATABRICKS_CONFIG_PROFILE="DEFAULT" # change to the profile name you chose
   ```

   **Option 2: Personal Access Token (PAT)**

   See the [PAT documentation](https://docs.databricks.com/aws/en/dev-tools/auth/pat#databricks-personal-access-tokens-for-workspace-users).

   ```bash
   # Add these to your .env file
   DATABRICKS_HOST="https://host.databricks.com"
   DATABRICKS_TOKEN="dapi_token"
   ```

   See the [Databricks SDK authentication docs](https://docs.databricks.com/aws/en/dev-tools/sdk-python#authenticate-the-databricks-sdk-for-python-with-your-databricks-account-or-workspace).

3. **Create and link an MLflow experiment to your app**

   Create an MLflow experiment to enable tracing and version tracking. This is automatically done by the `uv run quickstart` script.

   Create the MLflow experiment via the CLI:

   ```bash
   DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
   databricks experiments create-experiment /Users/$DATABRICKS_USERNAME/agents-on-apps
   ```

   Make a copy of `.env.example` to `.env` and update the `MLFLOW_EXPERIMENT_ID` in your `.env` file with the experiment ID you created. The `.env` file will be automatically loaded when starting the server.

   ```bash
   cp .env.example .env
   # Edit .env and fill in your experiment ID
   ```

   See the [MLflow experiments documentation](https://docs.databricks.com/aws/en/mlflow/experiments#create-experiment-from-the-workspace).

4. **Test your agent locally**

   Start up the agent server and chat UI locally:

   ```bash
   uv run start-app
   ```

   Query your agent via the UI (http://localhost:8000) or REST API:

   **Advanced server options:**

   ```bash
   uv run start-server --reload   # hot-reload the server on code changes
   uv run start-server --port 8001 # change the port the server listens on
   uv run start-server --workers 4 # run the server with multiple workers
   ```

   - Example streaming request:
     ```bash
     curl -X POST http://localhost:8000/invocations \
     -H "Content-Type: application/json" \
     -d '{ "input": [{ "role": "user", "content": "hi" }], "stream": true }'
     ```
   - Example non-streaming request:
     ```bash
     curl -X POST http://localhost:8000/invocations  \
     -H "Content-Type: application/json" \
     -d '{ "input": [{ "role": "user", "content": "hi" }] }'
     ```

## Advanced: on-behalf-of-user (OBO) auth for the LLM endpoint

By default the deployed app uses its **service principal** to call
`databricks-claude-sonnet-4` — the `databricks.yml` in this repo grants the
SP `CAN_QUERY` on that endpoint. That's the simplest pattern and the one
Industry Solution Accelerator adopters are used to.

If you'd rather have each request hit the endpoint **as the requesting
user** — useful for a shared demo where you want per-user quota and no SP
grants — swap to on-behalf-of (OBO) auth in two steps.

**1. Remove the SP grant** from `databricks.yml`. Delete the
`llm_endpoint` resource binding under `apps.agent_langgraph.resources`:

```yaml
# Delete this block for OBO:
- name: 'llm_endpoint'
  serving_endpoint:
    name: 'databricks-claude-sonnet-4'
    permission: 'CAN_QUERY'
```

**2. Instantiate `ChatDatabricks` per-request using the user's workspace
client** in `agent_server/agent.py`. Replace the module-level
`_llm = ChatDatabricks(...)` with a helper that pulls the forwarded user
token from `agent_server.utils.get_user_workspace_client()`:

```python
from databricks_langchain.chat_models import ChatDatabricks
from agent_server.utils import get_user_workspace_client

def _make_llm() -> ChatDatabricks:
    user_client = get_user_workspace_client()
    return ChatDatabricks(
        endpoint="databricks-claude-sonnet-4",
        client=user_client,
    )
```

Then call `_make_llm()` inside `streaming()` / `non_streaming()` and pass
the result into `build_graph(checkpointer=..., llm=_make_llm())`.

Every user of the deployed app must themselves have `CAN_QUERY` on the
serving endpoint. The `x-forwarded-access-token` header that
`get_user_workspace_client()` reads is only injected when the app is
deployed on Databricks Apps — for local development, keep the default
`ChatDatabricks()` that uses your `DATABRICKS_CONFIG_PROFILE`.

## Lakebase schema layout and ownership

This accelerator creates three Postgres schemas in the bundle-provisioned
Lakebase instance (`banking-agent-memory` by default):

| Schema | Owner after first successful run | Purpose |
|---|---|---|
| `agent_checkpoints` | App service principal | LangGraph state, written by `AsyncCheckpointSaver` in `agent_server/agent.py`. Schema name is controlled by the `CHECKPOINT_SCHEMA` env var (default `agent_checkpoints`). |
| `ai_chatbot` | Whoever runs the first migration | Chat UI's `User`/`Chat`/`Message` tables, written by Drizzle in the chatbot frontend. |
| `drizzle` | Same as above | Drizzle's own migration-tracking metadata. |

> **Important caveat for adopters.** If you run `uv run start-app` locally
> *before* deploying to Databricks Apps, the `ai_chatbot` and `drizzle`
> schemas will be owned by your user and the deployed app's service
> principal will fail its migration with a Postgres permission error. You
> have two options:
>
> 1. **Deploy first, local dev second.** Run `databricks bundle deploy &&
>    databricks bundle run agent_langgraph` once so the SP creates both
>    schemas. Then `uv run start-app` locally works because your user can
>    read/write schemas the SP owns (via `CAN_CONNECT_AND_CREATE` on the
>    database).
> 2. **Reset the schemas** before the first deploy if you've already run
>    locally: `DROP SCHEMA IF EXISTS ai_chatbot CASCADE; DROP SCHEMA IF
>    EXISTS drizzle CASCADE;` and then redeploy. The SP will recreate them.
>
> The LangGraph checkpointer (`agent_checkpoints`) does not hit this
> problem because it's a schema that neither user touches by default.

## Modifying your agent

See the [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/quickstart) for more information on how to edit your own agent.

Required files for hosting with MLflow `AgentServer`:

- `agent.py`: Contains your agent logic. Modify this file to create your custom agent. For example, you can [add agent tools](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-tool) to give your agent additional capabilities
- `start_server.py`: Initializes and runs the MLflow `AgentServer` with agent_type="ResponsesAgent". You don't have to modify this file for most common use cases, but can add additional server routes (e.g. a `/metrics` endpoint) here

**Common customization questions:**

**Q: Can I add additional files or folders to my agent?**
Yes. Add additional files or folders as needed. Ensure the script within `pyproject.toml` runs the correct script that starts the server and sets up MLflow tracing.

**Q: How do I add dependencies to my agent?**
Run `uv add <package_name>` (e.g., `uv add "mlflow-skinny[databricks]"`). See the [python pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-and-requirements).

**Q: Can I add custom tracing beyond the built-in tracing?**
Yes. This template uses MLflow's agent server, which comes with automatic tracing for agent logic decorated with `@invoke()` and `@stream()`. It also uses [MLflow autologging APIs](https://mlflow.org/docs/latest/genai/tracing/#one-line-auto-tracing-integrations) to capture traces from LLM invocations. However, you can add additional instrumentation to capture more granular trace information when your agent runs. See the [MLflow tracing documentation](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/).

**Q: How can I extend this example with additional tools and capabilities?**
This template can be extended by integrating additional MCP servers, Vector Search Indexes, UC Functions, and other Databricks tools. See the ["Agent Framework Tools Documentation"](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-tool).

## Evaluating your agent

Evaluate your agent by calling the invoke function you defined for the agent locally.

- Update your `evaluate_agent.py` file with the preferred evaluation dataset and scorers.

Run the evaluation using the evaluation script:

```bash
uv run agent-evaluate
```

After it completes, open the MLflow UI link for your experiment to inspect results.

## Deploying to Databricks Apps

0. **Create a Databricks App**:
   Ensure you have the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/tutorial) installed and configured.

   ```bash
   databricks apps create agent-langgraph
   ```

1. **Set up authentication to Databricks resources**

   For this example, you need to add an MLflow Experiment as a resource to your app. Grant the App's Service Principal (SP) permission to edit the experiment by clicking `edit` on your app home page. See the [Databricks Apps MLflow experiment documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/mlflow) for more information.

   To grant access to other resources like serving endpoints, genie spaces, UC Functions, and Vector Search Indexes, click `edit` on your app home page to grant the App's SP permission. See the [Databricks Apps resources documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/resources).

   For resources that are not supported yet, see the [Agent Framework authentication documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent#automatic-authentication-passthrough) for the correct permission level to grant to your app SP.

   **On-behalf-of (OBO) User Authentication**: Use `get_user_workspace_client()` from `agent_server.utils` to authenticate as the requesting user instead of the app service principal. See the [OBO authentication documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth?language=Streamlit#retrieve-user-authorization-credentials).

2. **Sync local files to your workspace**

   See the [Databricks Apps deploy documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy?language=Databricks+CLI#deploy-the-app).

   ```bash
   DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
   databricks sync . "/Users/$DATABRICKS_USERNAME/agent-langgraph"
   ```

3. **Deploy your Databricks App**

   See the [Databricks Apps deploy documentation](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy?language=Databricks+CLI#deploy-the-app).

   ```bash
   databricks apps deploy agent-langgraph --source-code-path /Workspace/Users/$DATABRICKS_USERNAME/agent-langgraph
   ```

4. **Query your agent hosted on Databricks Apps**

   Databricks Apps are _only_ queryable via OAuth token. You cannot use a PAT to query your agent. Generate an [OAuth token with your credentials using the Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/authentication#u2m-auth):

   ```bash
   databricks auth login --host <https://host.databricks.com>
   databricks auth token
   ```

   Send a request to the `/invocations` endpoint:

   - Example streaming request:

     ```bash
     curl -X POST <app-url.databricksapps.com>/invocations \
        -H "Authorization: Bearer <oauth token>" \
        -H "Content-Type: application/json" \
        -d '{ "input": [{ "role": "user", "content": "hi" }], "stream": true }'
     ```

   - Example non-streaming request:

     ```bash
     curl -X POST <app-url.databricksapps.com>/invocations \
        -H "Authorization: Bearer <oauth token>" \
        -H "Content-Type: application/json" \
        -d '{ "input": [{ "role": "user", "content": "hi" }] }'
     ```

For future updates to the agent, sync and redeploy your agent.

### FAQ

- For a streaming response, I see a 200 OK in the logs, but an error in the actual stream. What's going on?
  - This is expected behavior. The initial 200 OK confirms stream setup; streaming errors don't affect this status.
- When querying my agent, I get a 302 error. What's going on?
  - Use an OAuth token. PATs are not supported for querying agents.
