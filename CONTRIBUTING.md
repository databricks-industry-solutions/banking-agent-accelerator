# Contributing

Thanks for your interest in this Databricks Industry Solution Accelerator. The
project is maintained by the Databricks Field Engineering team.

## Contributor License Agreement (CLA)

By submitting a contribution to this repository, you certify that:

1. **You have the right to submit the contribution.**
   You created the code/content yourself, or you have the right to submit it
   under the project's license.

2. **You grant us a license to use your contribution.**
   You agree that your contribution will be licensed under the same terms as
   the rest of this project, and you grant the project maintainers the right
   to use, modify, and distribute your contribution as part of the project.

3. **You are not submitting confidential or proprietary information.**
   Your contribution does not include anything you don't have permission to
   share publicly.

If you are contributing on behalf of an organization, you confirm that you
have the authority to do so. You agree to confirm these terms in your pull
request. Any request that does not explicitly accept the terms will be
assumed to have accepted.

## Owners

See [`CODEOWNERS`](./CODEOWNERS). Pull requests require review from at least
one listed owner before merge.

## Repository structure

```
agent_app/                 Python LangGraph agent (FastAPI + Lakebase)
  agent_server/            Agent code, tools, evaluation
  e2e-chatbot-app-next/    Next.js + React chat UI (TypeScript monorepo)
  databricks.yml           Databricks Asset Bundle config
config/                    Workspace resource pointers
```

Per-component setup, run, and deploy instructions live in:

- [`agent_app/README.md`](./agent_app/README.md) — Python agent
- [`agent_app/agent_server/README.md`](./agent_app/agent_server/README.md) — agent internals
- [`agent_app/e2e-chatbot-app-next/README.md`](./agent_app/e2e-chatbot-app-next/README.md) — chat UI

## Development workflow

### Python agent (`agent_app/`)

```bash
cd agent_app
uv run quickstart        # one-time setup
uv run start-app         # local dev server
```

### Chat UI (`agent_app/e2e-chatbot-app-next/`)

```bash
cd agent_app/e2e-chatbot-app-next
./scripts/quickstart.sh  # one-time setup
npm run dev              # local dev server
```

### Deploy (Databricks Asset Bundle)

```bash
cd agent_app
databricks bundle deploy
databricks bundle run agent_langgraph
```

## Code style and linting

- **Python**: managed by `uv`. Format and type-check before submitting.
- **TypeScript**: linted and formatted with [Biome](https://biomejs.dev/) (not
  ESLint/Prettier).
  - `npm run lint` (auto-fixes)
  - `npm run format`

## Testing

- **Python**: `pytest` from `agent_app/`.
- **TypeScript**: `npm test` from `agent_app/e2e-chatbot-app-next/`
  (Playwright + MSW). Run tests locally before opening a PR.

## Pull request process

1. Open the PR against `main`.
2. Keep the change scoped — one logical change per PR.
3. Update the relevant `README.md` files when behavior changes.
4. Make sure linters and tests pass locally.
5. Request review from a `CODEOWNERS` member.
6. After review and CI pass, a maintainer will merge.

## Commit messages

Concise, imperative mood. Reference the affected component when useful, e.g.
`agent_server: pin checkpointer to dedicated schema`.
